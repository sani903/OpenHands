import shutil
import subprocess

# Path to the input .txt file
input_file = 'evaluation/swe_bench/data/instance_ids.txt'

# Path to the config.toml file
config_file = 'evaluation/swe_bench/config.toml'

# Path to the output.jsonl file
output_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/claude-3-5-sonnet-20240620_maxiter_30_N_v1.9-no-hint/output.jsonl'
counter = -1
# Read the input file line by line
with open(input_file, 'r') as f:
    for line in f:
        counter += 1
        if counter == 0:
            continue
        # Remove any leading/trailing whitespace
        new_string = line.strip()

        # Update the config.toml file
        with open(config_file, 'w') as config:
            config.write(f'selected_ids = [ "{new_string}" ]')

        # Run the evaluation command
        subprocess.run(
            ['./evaluation/swe_bench/scripts/run_infer.sh', 'llm.claude-3-5-sonnet'],
            check=True,
        )

        # Rename the output file
        new_output_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/claude-3-5-sonnet-20240620_maxiter_30_N_v1.9-no-hint/{new_string}_output.jsonl'
        shutil.move(output_file, new_output_file)

        print(f'Processed: {new_string}')

print('Script execution completed.')
