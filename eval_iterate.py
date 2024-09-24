import os
import subprocess

# Directory containing the .jsonl files
output_dir = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/claude-3-5-sonnet-20240620_maxiter_30_N_v1.9-no-hint/'
# Evaluation script path
eval_script = './evaluation/swe_bench/scripts/eval_infer.sh'


def evaluate_files():
    print('Current working directory:', os.getcwd())
    # Find all .jsonl files ending with output.jsonl in the specified directory and its subdirectories
    for root, _, files in os.walk(output_dir):
        for file in files:
            print(file)
            if file.endswith('output.jsonl'):
                file_path = os.path.join(root, file)
                print(f'Evaluating file: {file_path}')

                prefix = file.replace('output.jsonl', '')

                # Run the evaluation script for each file
                try:
                    subprocess.run([eval_script, file_path], check=True)
                    print(f'Evaluation complete for {file_path}')

                    # Rename README.md and report.json with the prefix
                    if os.path.exists('README.md'):
                        os.rename('README.md', f'{prefix}README.md')
                    if os.path.exists('report.json'):
                        os.rename('report.json', f'{prefix}report.json')

                except subprocess.CalledProcessError as e:
                    print(f'Error evaluating {file_path}: {e}')
                print('----------------------------------------')

    print('All evaluations completed.')


if __name__ == '__main__':
    evaluate_files()
