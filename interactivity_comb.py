import glob
import os


def process_files(folder_path):
    # Initialize counters
    total_files = 0
    gold_1_count = 0
    gold_0_count = 0
    matching_count = 0
    gold_0_questions_0_count = 0
    gold_1_questions_1_count = 0

    # Get all question files
    question_files = glob.glob(os.path.join(folder_path, 'test_*_questions.txt'))

    for question_file in question_files:
        total_files += 1

        # Get the corresponding gold file
        base_name = os.path.splitext(question_file)[0]
        gold_file = base_name.replace('_questions', '_gold') + '.txt'

        if os.path.exists(gold_file):
            # Read question file
            with open(question_file, 'r') as f:
                question_value = f.read().strip()

            # Read gold file
            with open(gold_file, 'r') as f:
                gold_content = f.read().strip()
                gold_value = gold_content.split(':')[-1].strip()

            # Update counts
            if gold_value == '1':
                gold_1_count += 1
            elif gold_value == '0':
                gold_0_count += 1

            if question_value == gold_value:
                matching_count += 1
                if question_value == '0':
                    gold_0_questions_0_count += 1
                elif question_value == '1':
                    gold_1_questions_1_count += 1

    return {
        'total_files': total_files,
        'gold_1_count': gold_1_count,
        'gold_0_count': gold_0_count,
        'matching_count': matching_count,
        'gold_0_questions_0_count': gold_0_questions_0_count,
        'gold_1_questions_1_count': gold_1_questions_1_count,
    }


# Usage
folder_path = 'evaluation/evaluation_outputs/outputs/swe-bench-lite/CodeActAgent/deepseek-chat_maxiter_4_N_v1.9-no-hint/'
results = process_files(folder_path)

print(f"Total files processed: {results['total_files']}")
print(f"Number of gold files with value 1: {results['gold_1_count']}")
print(f"Number of gold files with value 0: {results['gold_0_count']}")
print(f"Number of matching values: {results['matching_count']}")
print(
    f"Number of times both gold and questions are 0: {results['gold_0_questions_0_count']}"
)
print(
    f"Number of times both gold and questions are 1: {results['gold_1_questions_1_count']}"
)
