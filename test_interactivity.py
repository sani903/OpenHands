import os
import shutil
import subprocess

# Path to the input .txt file
input_file = 'evaluation/swe_bench/data/subset_ids.txt'

# Path to the config.toml file
config_file = 'evaluation/swe_bench/config.toml'

# Path to the output.jsonl file
gold_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_4_N_v1.9-no-hint/test_gold.txt'
questions_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_4_N_v1.9-no-hint/question.txt'
output_file = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_4_N_v1.9-no-hint/output.jsonl'
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
        counter += 1
        # Remove any leading/trailing whitespace
        #        remove_docker_images()
        new_string = line.strip()
        new_questions_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_4_N_v1.9-no-hint/test_{new_string}_questions.txt'
        new_output_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_4_N_v1.9-no-hint/test_{new_string}_output.jsonl'
        new_gold_file = f'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_4_N_v1.9-no-hint/test_{new_string}_gold.txt'
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
                [
                    './evaluation/swe_bench/scripts/test_interactivity.sh',
                    'llm.deepseek-chat',
                ],
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
        shutil.move(questions_file, new_questions_file)
        shutil.move(gold_file, new_gold_file)
        print(f'Processed: {new_string}')

print('Script execution completed.')
