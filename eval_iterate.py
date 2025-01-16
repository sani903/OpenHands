import os
import subprocess

# Directory containing the .jsonl files
output_dir = '/Users/sanid/Desktop/Interactivity/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench-test/CodeActAgent/claude-3-5-haiku-20241022_maxiter_30_N_v0.16.0-no-hint-run_1/'  # Evaluation script path
# eval_script = './evaluation/swe_bench/scripts/eval_infer.sh'

os.environ['ALLHANDS_API_KEY'] = 'ah-73a36a7d-a9f4-4f52-aa85-215abc90a96f'
os.environ['RUNTIME'] = 'remote'
os.environ['SANDBOX_REMOTE_RUNTIME_API_URL'] = 'https://runtime.eval.all-hands.dev'
os.environ['EVAL_DOCKER_IMAGE_PREFIX'] = (
    'us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images'
)


def evaluate_files():
    print('Current working directory:', os.getcwd())
    # Find all .jsonl files ending with output.jsonl in the specified directory and its subdirectories
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file == 'haiku_fin_hidden.jsonl':
                print(file)
                # if file.startswith('interact') and file.endswith('.jsonl'):
                # if file.startswith('interact') and file.endswith('output.jsonl'):
                file_path = os.path.join(root, file)
                print(f'Evaluating file: {file_path}')

                prefix = file.replace('.jsonl', '')

                # Run the evaluation script for each file
                try:
                    subprocess.run(
                        [
                            './evaluation/benchmarks/swe_bench/scripts/eval_infer_remote.sh',
                            file_path,
                        ],
                        check=True,
                        env=os.environ,  # Pass the environment variables
                    )
                    print(f'Evaluation complete for {file_path}')

                    # Rename README.md and report.json with the prefix
                    if os.path.exists(f'{output_dir}README.md'):
                        print('Found README')
                        os.rename(f'{output_dir}README.md', f'{prefix}README.md')
                    if os.path.exists(f'{output_dir}report.json'):
                        print('FOUND REPORT')
                        os.rename(f'{output_dir}report.json', f'{prefix}report.json')

                except subprocess.CalledProcessError as e:
                    print(f'Error evaluating {file_path}: {e}')
                print('----------------------------------------')

    print('All evaluations completed.')


if __name__ == '__main__':
    evaluate_files()
