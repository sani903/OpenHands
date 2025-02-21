import asyncio
import json
import os
import re
import tempfile
from typing import Any

import openai
import pandas as pd
import toml

import openhands.agenthub
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    get_default_sandbox_config_for_eval,
    get_metrics,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    AppConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue
import copy
from openhands.event.event import EventSource

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
USE_INSTANCE_IMAGE = os.environ.get('USE_INSTANCE_IMAGE', 'true').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

client = openai.OpenAI(
    api_key=os.environ['LITELLM_API_KEY'],
    base_url='https://cmu.litellm.ai',
)


class FakeUser:
    def __init__(self, issue, hints, files):
        self.system_message = f"""
        You are a GitHub user reporting an issue. Here are the details of your issue and environment:

        Issue: {issue}

        Hints: {hints}

        Files relative to your current directory: {files}

        Your task is to respond to questions from a coder who is trying to solve your issue. The coder has a summarized version of the issue you have. Follow these rules:
        1. If the coder asks a question that is directly related to the information in the issue you have, provide that information.
        2. Always stay in character as a user reporting an issue, not as an AI assistant.
        3. Keep your responses concise and to the point.
        4. The coder has limited turns to solve the issue. Do not interact with the coder beyond 3 turns.

        Respond with "I don't have that information" if the question is unrelated or you're unsure.
        """
        self.chat_history = [{'role': 'system', 'content': self.system_message}]
        self.turns = 0

    def generate_reply(self, question):
        if self.turns > 3:
            return 'Please continue working on the task. Do NOT ask for more help.'
        self.chat_history.append({'role': 'user', 'content': question.content})
        response = client.chat.completions.create(
            model='neulab/gpt-4o-2024-05-13', messages=self.chat_history
        )
        reply = response.choices[0].message.content
        self.chat_history.append({'role': 'assistant', 'content': reply})
        self.turns += 1
        return reply


USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
USE_INSTANCE_IMAGE = os.environ.get('USE_INSTANCE_IMAGE', 'false').lower() == 'true'

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': lambda state: fake_user_response(state),
    'CodeActSWEAgent': lambda state: fake_user_response(state),
}


def fake_user_response(state: State) -> str:
    last_agent_message = state.get_last_agent_message()

    if last_agent_message:
        return fake_user.generate_reply(last_agent_message)
    else:
        return 'Please continue working on the task.'


AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.\n',
    'CodeActSWEAgent': 'When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.\n',
}


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata):
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    # Prepare instruction
    # Instruction based on Anthropic's official trajectory
    # https://github.com/eschluntz/swe-bench-experiments/tree/main/evaluation/verified/20241022_tools_claude-3-5-sonnet-updated/trajs
    instruction = (
        '<uploaded_files>\n'
        f'/workspace/{workspace_dir_name}\n'
        '</uploaded_files>\n'
        f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following issue description:\n\n"
        f'<issue_description>\n'
        f'{instance.problem_statement}\n'
        '</issue_description>\n\n'
        'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
        "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
        'I have not provided all the necessary details about the issue and I have some hidden details that are helpful. Please ask me specifc questions using non-code commands to gather the relevant information that I have to help you solve the issue. Ensure you have all the details you require to solve the issue.\n'
        'You have a limited number of turns. Do NOT interact with me more than three times so as to maximize the number of turns you have to work on the solution.\n'
        'Follow these steps to resolve the issue:\n'
        '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
        '2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error\n'
        '3. Edit the sourcecode of the repo to resolve the issue\n'
        '4. Rerun your reproduce script and confirm that the error is fixed!\n'
        '5. Think about edgecases and make sure your fix handles them as well\n'
        "Your thinking should be thorough and so it's fine if it's very long.\n"
    )

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\n'
            'You SHOULD NEVER attempt to browse the web. '
            '</IMPORTANT!>\n'
        )
    return instruction


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    SWE_BENCH_CONTAINER_IMAGE = 'ghcr.io/opendevin/eval-swe-bench:full-v1.2.1'
    if USE_INSTANCE_IMAGE:
        # We use a different instance image for the each instance of swe-bench eval
        base_container_image = get_instance_docker_image(instance['instance_id'])
        logger.info(
            f'Using instance container image: {base_container_image}. '
            f'Please make sure this image exists. '
            f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
        )
    else:
        base_container_image = SWE_BENCH_CONTAINER_IMAGE
        logger.info(f'Using swe-bench container image: {base_container_image}')

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    agent_config = AgentConfig(
        codeact_enable_jupyter=False,
        codeact_enable_browsing=RUN_WITH_BROWSING,
        codeact_enable_llm_editor=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    obs: CmdOutputObservation

    # Set instance id
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0, f'Failed to export SWE_INSTANCE_ID: {str(obs)}'
    )

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    if USE_INSTANCE_IMAGE:
        # inject the init script
        script_dir = os.path.dirname(__file__)

        # inject the instance info
        action = CmdRunAction(command='mkdir -p /swe_util/eval_data/instances')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to create /swe_util/eval_data/instances: {str(obs)}',
        )

        swe_instance_json_name = 'swe-bench-instance.json'
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct the full path for the desired file name within the temporary directory
            temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
            # Write to the file with the desired name within the temporary directory
            with open(temp_file_path, 'w') as f:
                if not isinstance(instance, dict):
                    json.dump([instance.to_dict()], f)
                else:
                    json.dump([instance], f)

            # Copy the file to the desired location
            runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        # inject the instance swe entry
        runtime.copy_to(
            str(os.path.join(script_dir, 'scripts/setup/instance_swe_entry.sh')),
            '/swe_util/',
        )
        action = CmdRunAction(command='cat ~/.bashrc')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to cat ~/.bashrc: {str(obs)}')

        action = CmdRunAction(command='source ~/.bashrc')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if isinstance(obs, ErrorObservation):
            logger.error(f'Failed to source ~/.bashrc: {str(obs)}')
        assert_and_raise(obs.exit_code == 0, f'Failed to source ~/.bashrc: {str(obs)}')

        action = CmdRunAction(command='source /swe_util/instance_swe_entry.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/instance_swe_entry.sh: {str(obs)}',
        )

    else:
        action = CmdRunAction(command='source /swe_util/swe_entry.sh')
        action.set_hard_timeout(1800)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/swe_entry.sh: {str(obs)}',
        )
    try:
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
        )
    # for custom SWE-Bench Verified Dataset
    except Exception:
        alt_name = workspace_dir_name.rsplit('.0', 1)[0]
        action = CmdRunAction(command=f'cd /workspace/{alt_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to cd to /workspace/{alt_name}: {str(obs)}',
        )

    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')

    action = CmdRunAction(command='which python')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0 and 'testbed' in obs.content,
        f'Expected to find python interpreter from testbed, but got: {str(obs)}',
    )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    try:
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if obs.exit_code == -1:
            # The previous command is still running
            # We need to kill previous command
            logger.info('The previous command is still running, trying to kill it...')
            action = CmdRunAction(command='C-c')
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            # Then run the command again
            action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
            action.set_hard_timeout(600)
        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
        )
    except Exception:
        alt_name = workspace_dir_name.rsplit('.0', 1)[0]
        action = CmdRunAction(command=f'cd /workspace/{alt_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if obs.exit_code == -1:
            # The previous command is still running
            # We need to kill previous command
            logger.info('The previous command is still running, trying to kill it...')
            action = CmdRunAction(command='C-c')
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            # Then run the command again
            action = CmdRunAction(command=f'cd /workspace/{alt_name}')
            action.set_hard_timeout(600)
            assert_and_raise(
                isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
                f'Failed to cd to /workspace/{alt_name}: {str(obs)}',
            )

    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git config --global core.pager "": {str(obs)}',
    )

    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {instance["base_commit"]}'
        )
        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                git_patch = obs.content.strip()
                break
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    config = get_config(instance, metadata)
    global fake_user
    original_issue = instance.original_issue
    issue = str(original_issue)
    fake_user = FakeUser(issue=issue, hints=instance.hints_text, files=instance.files)
    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')
    # Increase resource_factor with increasing attempt_id
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        initialize_runtime(runtime, instance)

        instruction = get_instruction(instance, metadata)

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=MessageAction(content=instruction),
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )
        # if fatal error, throw EvalError to trigger re-run

        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)
        # ======= THIS IS SWE-Bench specific =======
        # Get git patch
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
        # --- Verification Step ---
        # Only send a verification message if the git patch is nonempty.
        if git_patch.strip():
            verification_message = (
                f'Here is the generated patch:\n\n{git_patch}\n\n'
                'Please verify how many items from this checklist are satisfied:\n\n'
                f'{state.checklist}'
            )
            verification_action = MessageAction(content=verification_message)
            verification_action._source = EventSource.USER
            state.event_stream.add_event(verification_action, EventSource.USER)
            verification_config = copy.deepcopy(config)
            verification_config.max_iterations = 1
            verification_state = asyncio.run(
                run_controller(
                    config=verification_config,
                    initial_user_action=verification_action,
                    runtime=runtime,
                    fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                        metadata.agent_class
                    ],
                )
            )
            verification_response = verification_state.get_last_agent_message()
            if verification_response:
                match = re.search(
                    r'(\d+)\s+checklist items? fulfilled',
                    verification_response.content,
                    re.IGNORECASE,
                )
                if match:
                    items_completed = int(match.group(1))
                    logger.info(
                        f'Verification: {items_completed} checklist items fulfilled.'
                    )
                else:
                    items_completed = 0
                    logger.warning(
                        'Verification response did not match expected format.'
                    )
    finally:
        runtime.close()
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'git_patch': git_patch,
        'verification_result': items_completed,
    }

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)
    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if 'selected_ids' in data:
                selected_ids = data['selected_ids']
                logger.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                subset = dataset[dataset[filter_column].isin(selected_ids)]
                logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                return subset
    skip_ids = os.environ.get('SKIP_IDS', '').split(',')
    if len(skip_ids) > 0:
        logger.info(f'Filtering {len(skip_ids)} tasks from "SKIP_IDS"...')
        return dataset[~dataset[filter_column].isin(skip_ids)]
    return dataset


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default='evaluation/benchmarks/swe_bench/data/full_summaries.xlsx',
        help='Path to the CSV file containing the dataset',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    #    dataset = load_dataset(args.dataset, split=args.split)
    csv_filepath = args.csv_file
    dataset = pd.read_excel(csv_filepath)
    logger.info(f'Loaded dataset from {csv_filepath}')
    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
    )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)
    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    print(f'### OUTPUT FILE: {output_file} ###')
    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)

    if len(instances) > 0 and not isinstance(
        instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            instances[col] = instances[col].apply(lambda x: str(x))

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=120 * 60,  # 2 hour PER instance should be more than enough
        max_retries=5,
    )
