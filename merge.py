import glob
import json
import os

# Specify the directory path
directory_path = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/'

# Pattern to match files starting with "base" and ending with ".jsonl"
file_pattern = os.path.join(directory_path, 'hidden*.jsonl')

# Output file name
output_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/Meta-Llama-3.1-70B-Instruct_maxiter_30_N_v1.9-no-hint/concatenated_hidden.jsonl'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Iterate over all matching files
    for filename in glob.glob(file_pattern):
        with open(filename, 'r') as infile:
            # Read each line from the input file and write it to the output file
            for line in infile:
                # Optionally, you can validate each line as valid JSON
                try:
                    json.loads(line)
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f'Skipping invalid JSON in file {filename}')

print(f'Concatenation complete. Output saved to {output_file}')
