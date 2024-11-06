import sys 
import os
import shutil
import subprocess

input_file = 'evaluation/swe_bench/data/subset_ids.txt'

config_file = 'evaluation/swe_bench/config.toml'
output_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_30_N_v1.9-no-hint/output.jsonl'
turns_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_30_N_v1.9-no-hint/output_turns.txt'




counter = 0
all_combined = ""
with open(input_file, 'r') as f:
    for line in f:
        new_string = line.strip()
        all_combined+="'"
        all_combined+=new_string
        all_combined+="', "

all_combined= all_combined[:-2]
print(all_combined)
with open(config_file, 'w') as config:
    config.write(f'selected_ids = [{all_combined}]')
os.environ['ALLHANDS_API_KEY'] = "ah-73a36a7d-a9f4-4f52-aa85-215abc90a96f"
os.environ['RUNTIME'] = "remote"
os.environ['SANDBOX_REMOTE_RUNTIME_API_URL'] = "https://runtime.eval.all-hands.dev"
os.environ['EVAL_DOCKER_IMAGE_PREFIX'] = "us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
try:
    subprocess.run(
        [
            './evaluation/swe_bench/scripts/run_infer.sh',
            'llm.llama',
            'HEAD',
            'CodeActAgent',
            '300',
            '30',
            '16',
            '"princeton-nlp/SWE-bench"',
            'test'
        ],
        check=True,
        env=os.environ  # Pass the environment variables
    )
except subprocess.CalledProcessError as e:
    print(f'Subprocess error: {e}')
    print(f'Return code: {e.returncode}')
    print(f'Output: {e.output}')
except FileNotFoundError as e:
    print(f'File not found error: {e}')
except PermissionError as e:
    print(f'Permission error: {e}')
except Exception as e:
    print(f'Unexpected error: {e}')

print('Script execution completed.')
