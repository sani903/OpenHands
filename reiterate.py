import json
import os
import shutil
import subprocess

# Directory containing the .jsonl files
directory_path = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/'


# Function to process a single file
def process_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                instance_id = data.get('instance_id', '')
                git_patch = data.get('test_result', {}).get('git_patch', '')
                if git_patch == '':
                    return instance_id
            except json.JSONDecodeError:
                continue
    return None


# Main processing loop
for filename in os.listdir(directory_path):
    if filename.startswith('hidden') and filename.endswith('.jsonl'):
        file_path = os.path.join(directory_path, filename)
        instance_id = process_file(file_path)

        if instance_id:
            # Set up paths
            #    input_file = 'evaluation/swe_bench/data/subset_ids.txt'
            config_file = 'evaluation/swe_bench/config.toml'
            output_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/output.jsonl'

            # Update config.toml
            with open(config_file, 'w') as config:
                config.write(f'selected_ids = [ "{instance_id}" ]')

            # Run evaluation command
            try:
                subprocess.run(
                    ['./evaluation/swe_bench/scripts/run_infer.sh', 'llm.llama-3-1'],
                    check=True,
                )
            except subprocess.CalledProcessError:
                print(f'Error processing: {instance_id}')
                continue

            # Rename output file
            new_output_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/hidden_{instance_id}_output.jsonl'
            #   if not os.path.exists(new_output_file):
            shutil.move(output_file, new_output_file)
            print(f'Processed: {instance_id}')
        #    else:
        #        print(f"File already exists: {instance_id}")
