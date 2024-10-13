import json
import os

# Directory containing the JSON files
directory = os.path.dirname(os.path.abspath(__file__))

# Initialize variables to accumulate values
total_resolved = 0
total_unresolved = 0
total_empty_patch = 0
total_error = 0

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith('ablation') and filename.endswith('report.json'):
        file_path = os.path.join(directory, filename)

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract and accumulate the required values
        total_resolved += data.get('resolved_instances', 0)
        if data.get('resolved_instances', 0) != 0:
            print(filename)
        total_unresolved += data.get('unresolved_instances', 0)
        total_empty_patch += data.get('empty_patch_instances', 0)
        total_error += data.get('error_instances', 0)

# Print the accumulated values
print(f'Total resolved instances: {total_resolved}')
print(f'Total unresolved instances: {total_unresolved}')
print(f'Total empty patch instances: {total_empty_patch}')
print(f'Total error instances: {total_error}')
