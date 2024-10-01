import os
import shutil
import subprocess

# Path to the input .txt file
input_file = 'evaluation/swe_bench/data/subset_ids.txt'

# Path to the config.toml file
config_file = 'evaluation/swe_bench/config.toml'

# Path to the output.jsonl file
output_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/output.jsonl'
# skip = True
# Read the input file line by line
counter = 0
with open(input_file, 'r') as f:
    for line in f:
        #        counter+=1
        # Remove any leading/trailing whitespace
        new_string = line.strip()
        file_path = os.path.join('directory_path', 'filename.txt')
        new_output_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/interact_{new_string}_output.jsonl'
        if os.path.exists(new_output_file):
            print(new_string)
            continue
        #       if skip:
        #          continue
        #        if 'matplotlib' in new_string or 'pytest' in new_string or 'sphinx' in new_string or 'scikit-learn' in new_string:
        #            continue
        #        if skip:
        #            continue

        # Update the config.toml file

        with open(config_file, 'w') as config:
            config.write(f'selected_ids = [ "{new_string}" ]')
        try:
            # Run the evaluation command
            subprocess.run(
                ['./evaluation/swe_bench/scripts/run_infer.sh', 'llm.llama-3-1'],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f'Subprocess error: {e}')
            print(f'Return code: {e.returncode}')
            print(f'Output: {e.output}')
            continue
        except FileNotFoundError as e:
            print(f'File not found error: {e}')
            continue
        except PermissionError as e:
            print(f'Permission error: {e}')
            continue
        except Exception as e:
            print(f'Unexpected error: {e}')
            continue
        # Rename the output file
        shutil.move(output_file, new_output_file)

        print(f'Processed: {new_string}')

print('Script execution completed.')
