import sys 
import os
import shutil
import subprocess

# Path to the input .txt file
input_file = 'evaluation/swe_bench/data/subset_ids.txt'

# Path to the config.toml file
config_file = 'evaluation/swe_bench/config.toml'
current_directory = os.getcwd()

# Add the current directory to sys.path
sys.path.append(current_directory)

# Optionally, you can also modify the PATH environment variable
os.environ['PATH'] += os.pathsep + current_directory
# Path to the output.jsonl file
output_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_30_N_v1.9-no-hint/output.jsonl'
turns_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_30_N_v1.9-no-hint/output_turns.txt'
# skip = True
# Read the input file line by line


def remove_docker_images():
    # Command to list docker images
    list_command = (
        "docker images ghcr.io/all-hands-ai/runtime --format '{{.Repository}}:{{.Tag}}'"
    )

    try:
        # Execute the list command and capture the output
        result = subprocess.run(
            list_command, shell=True, check=True, capture_output=True, text=True
        )
        image_list = result.stdout.strip().split('\n')

        # If there are images to remove
        if image_list and image_list[0]:
            # Command to remove the images
            remove_command = f"docker rmi {' '.join(image_list)}"

            # Execute the remove command
            subprocess.run(remove_command, shell=True, check=True)
            print(f'Successfully removed {len(image_list)} images.')
        else:
            print('No images found to remove.')

    except subprocess.CalledProcessError as e:
        print(f'An error occurred: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


counter = 0
with open(input_file, 'r') as f:
    for line in f:
        #        counter+=1
        # Remove any leading/trailing whitespace
        #        remove_docker_images()
        new_string = line.strip()
        new_output_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_30_N_v1.9-no-hint/hidden_{new_string}_output.jsonl'
        new_turns_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_30_N_v1.9-no-hint/hidden_{new_string}_turns.txt'
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
        os.environ['ALLHANDS_API_KEY'] = "ah-73a36a7d-a9f4-4f52-aa85-215abc90a96f"
        os.environ['RUNTIME'] = "remote"
        os.environ['SANDBOX_REMOTE_RUNTIME_API_URL'] = "https://runtime.eval.all-hands.dev"
        os.environ['EVAL_DOCKER_IMAGE_PREFIX'] = "us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"

        try:
            subprocess.run(
                [
                    './evaluation/swe_bench/scripts/run_infer.sh',
                    'llm.deepseek-chat',
                ],
            check=True,
            env=os.environ  # Pass the environment variables
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
        shutil.move(turns_file, new_turns_file)
        print(f'Processed: {new_string}')

print('Script execution completed.')
