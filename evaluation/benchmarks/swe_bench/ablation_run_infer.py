import asyncio
import json
import os
import tempfile
from typing import Any

import openai
import pandas as pd
import toml

import openhands.agenthub
from evaluation.swe_bench.prompt import CODEACT_SWE_PROMPT
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.runtime import Runtime
from openhands.runtime.utils.shutdown_listener import sleep_if_should_continue

client = openai.OpenAI(
    api_key=os.environ['LITELLM_API_KEY'],
    base_url='https://cmu.litellm.ai',
)


class FakeUser:
    def __init__(self, issue, hidden_details):
        self.system_message = f"""
        You are a GitHub user reporting an issue. Here are the details of your issue and environment:

        Issue: {issue}

        Your task is to respond to questions from a coder who is trying to solve your issue. Follow these rules:
        1. If the coder asks a question related to information that you have, respond to that question based on the issue description.
        2. If the question is outside the scope of the information you already have do not attempt to find that information.
        3. Always stay in character as a user reporting an issue, not as an AI assistant.
        4. Keep your responses concise and to the point.

        Respond with "I don't have that information" if the question is unrelated or you're unsure. Do NOT start assisting the coder by doing any solving yourself. You must only answer questions based on the issue description you have.
        """
        self.chat_history = [{'role': 'system', 'content': self.system_message}]

    def generate_reply(self, question):
        self.chat_history.append({'role': 'user', 'content': question})

        response = client.chat.completions.create(
            model='neulab/claude-3-5-sonnet-20240620', messages=self.chat_history
        )

        reply = response.choices[0].message.content
        self.chat_history.append({'role': 'assistant', 'content': reply})

        return reply


USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
USE_INSTANCE_IMAGE = os.environ.get('USE_INSTANCE_IMAGE', 'false').lower() == 'true'

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': lambda state: fake_user_response(state),
    'CodeActSWEAgent': lambda state: fake_user_response(state),
}


def fake_user_response(state: State) -> str:
    last_agent_message = None
    events = list(state.history.get_events())
    for event in reversed(events):
        if isinstance(event, MessageAction) and event.source == 'agent':
            last_agent_message = event.content
            break

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
    if metadata.agent_class == 'CodeActSWEAgent':
        instruction = (
            'We are currently solving the following issue within our repository. Here is the issue text:\n'
            '--- BEGIN ISSUE ---\n'
            f'{instance.original_issue}\n'
            '--- END ISSUE ---\n\n'
        )
        if USE_HINT_TEXT and instance.hints_text:
            instruction += (
                f'--- BEGIN HINTS ---\n{instance.hints_text}\n--- END HINTS ---\n'
            )
        instruction += CODEACT_SWE_PROMPT.format(workspace_dir_name=workspace_dir_name)
    else:
        # Testing general agents
        instruction = (
            f'Please fix the following issue for the repository in /workspace/{workspace_dir_name}.\n'
            'Environment has been set up for you to start working. You may assume all necessary tools are installed.\n\n'
            '# Problem Statement\n'
            f'{instance.original_issue}\n\n'
        )
        if USE_HINT_TEXT and instance.hints_text:
            instruction += f'# Hints\n{instance.hints_text}\n\n'
        instruction += (
            'You must first ask me questions about potential clarifications or if you need additional information. Please ask me specifc questions using non-code commands to help you solve the issue.\n'
            'You should NOT modify any existing test case files. If needed, you can add new test cases in a NEW file to reproduce the issue.\n'
            'You SHOULD INCLUDE PROPER INDENTATION in your edit commands.\n'
        )

    # NOTE: You can actually set slightly different instruction for different agents
    instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]
    return instruction


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name


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

    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_budget_per_task=4,
        max_iterations=metadata.max_iterations,
        runtime=os.environ.get('RUNTIME', 'eventstream'),
        sandbox=SandboxConfig(
            base_container_image=base_container_image,
            enable_auto_lint=True,
            use_host_network=False,
            # large enough timeout, since some testcases take very long to run
            timeout=300,
            api_key=os.environ.get('ALLHANDS_API_KEY', None),
            remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
            keep_remote_runtime_alive=False,
        ),
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(metadata.llm_config)
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
    # workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    obs: CmdOutputObservation

    # Set instance id
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc"""
    )
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if obs.exit_code != 0:
        logger.error(f'Command failed with exit code {obs.exit_code}: {obs.content}')
        # Handle the error appropriately, maybe by raising a custom exception
        raise RuntimeError(f'Failed to initialize runtime: {obs.content}')
    # assert obs.exit_code == 0

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if obs.exit_code != 0:
        logger.error(f'Command failed with exit code {obs.exit_code}: {obs.content}')
        # Handle the error appropriately, maybe by raising a custom exception
        raise RuntimeError(f'Failed to initialize runtime: {obs.content}')
    #    assert obs.exit_code == 0

    if USE_INSTANCE_IMAGE:
        # inject the init script
        script_dir = os.path.dirname(__file__)

        # inject the instance info
        action = CmdRunAction(command='mkdir -p /swe_util/eval_data/instances')
        action.timeout = 600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            obs.exit_code == 0
        ), f'Failed to create /swe_util/eval_data/instances: {obs.content}'

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
        action.timeout = 600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        action = CmdRunAction(command='source ~/.bashrc')
        action.timeout = 600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        action = CmdRunAction(command='source /swe_util/instance_swe_entry.sh')
        action.timeout = 3600
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0
    else:
        action = CmdRunAction(command='source /swe_util/swe_entry.sh')
        action.timeout = 1800
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            obs.exit_code == 0
        ), f'Failed to source /swe_util/swe_entry.sh: {obs.content}'

    action = CmdRunAction(command='cd /workspace/')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    action = CmdRunAction(command='cd "$(ls | head -n 1)"')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    action = CmdRunAction(command='git reset --hard')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

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
    # workspace_dir_name = _get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command='cd /workspace/')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    action = CmdRunAction(command='cd "$(ls | head -n 1)"')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='git config --global core.pager ""')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='git add -A')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {instance["base_commit"]}',
            keep_prompt=False,
        )
        action.timeout = 600 + 100 * n_retries
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
            raise ValueError(f'Unexpected observation type: {type(obs)}')

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(instance, metadata)
    global fake_user
    # df = pd.read_csv("data/fake_user_issues_under_0.csv")
    # issue = df.loc[df['instance_id'] == instance["instance_id"], 'issue'].iloc[0]
    # hidden_details_merged = df.loc[df['instance_id'] == instance["instance_id"], 'hidden_details'].iloc[0]
    original_issue = instance.original_issue
    hidden_details_merged = instance.hidden_details
    print(f"""
    These are the hidden_details: {hidden_details_merged}
    """)
    logger.info(f'These are the hidden_details: {hidden_details_merged}')
    delimiter = '|||'
    hidden_details_split = hidden_details_merged.split(delimiter)
    fake_user = FakeUser(issue=original_issue, hidden_details=hidden_details_split)
    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')
    runtime = create_runtime(config)
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
        # ======= THIS IS SWE-Bench specific =======
        # Get git patch
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close()  # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'git_patch': git_patch,
    }

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    histories = [event_to_dict(event) for event in state.history.get_events()]
    metrics = state.metrics.get() if state.metrics else None
    num_turns = sum(1 for _ in state.history.get_events()) if state else 0
    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        llm_completions=state.extra_data.get('llm_completions', []),
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        num_turns=num_turns,
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
        default='evaluation/swe_bench/data/transformed_verified_underspecified_0.csv',
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
    dataset = pd.read_csv(csv_filepath)
    logger.info(f'Loaded dataset from {csv_filepath}')
    swe_bench_tests = filter_dataset(dataset, 'instance_id')

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)
    if hasattr(_agent_cls, 'system_message'):
        details['system_message'] = _agent_cls.system_message
    if hasattr(_agent_cls, 'in_context_example'):
        details['in_context_example'] = _agent_cls.in_context_example

    metadata = make_metadata(
        llm_config,
        'swe-bench-lite',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)

    if len(instances) > 0 and not isinstance(
        instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            instances[col] = instances[col].apply(lambda x: str(x))

    run_evaluation(
        instances, metadata, output_file, args.eval_num_workers, process_instance
    )
