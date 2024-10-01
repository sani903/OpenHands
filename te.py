import os
import subprocess
import time

# Directory containing the .jsonl files
output_dir = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/'
# Evaluation script path
eval_script = './evaluation/swe_bench/scripts/eval_infer.sh'


def evaluate_files():
    print('Current working directory:', os.getcwd())
    # Find all .jsonl files ending with output.jsonl in the specified directory and its subdirectories
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('output.jsonl'):
                file_path = os.path.join(root, file)
                print(f'Evaluating file: {file_path}')

                prefix = file.replace('output.jsonl', '')

                # Run the evaluation script for each file
                try:
                    subprocess.run([eval_script, file_path], check=True)
                    print(f'Evaluation complete for {file_path}')

                    # Wait for a short time to ensure files are created
                    time.sleep(2)

                    # Rename README.md and report.json with the prefix
                    for filename in ['README.md', 'report.json']:
                        old_path = os.path.join(output_dir, filename)
                        new_path = os.path.join(output_dir, f'{prefix}{filename}')
                        if os.path.exists(old_path):
                            print(f'Found {filename}')
                            try:
                                os.rename(old_path, new_path)
                                print(f'Renamed {filename} to {prefix}{filename}')
                            except OSError as e:
                                print(f'Error renaming {filename}: {e}')
                        else:
                            print(f'{filename} not found in {output_dir}')

                except subprocess.CalledProcessError as e:
                    print(f'Error evaluating {file_path}: {e}')
                print('----------------------------------------')

    print('All evaluations completed.')


if __name__ == '__main__':
    evaluate_files()
