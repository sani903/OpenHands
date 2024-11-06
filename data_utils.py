import json
import os
import re


def extract_conversation(history):
    conversation = []
    for step in history:
        if step['source'] == 'user':
            conversation.append({'role': 'user', 'content': step['message']})
        elif step['source'] == 'agent':
            conversation.append({'role': 'assistant', 'content': step['message']})
    return conversation

def load_data(file_path):
    data_list = []
    directory_name = os.path.dirname(file_path)
    print('Directory name: ', directory_name)
    # with open(os.path.join(directory_name, 'report.json'), 'r') as report_file:
    #     data = json.load(report_file)
    #     resolved_ids = data.get('resolved_ids', [])
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            instance = data.get('instance_id')
            problem_statement = data.get('instance', {}).get('problem_statement')
            # resolved = 1 if instance in resolved_ids else 0
            conversation = extract_conversation(data.get('history', []))
            data_list.append((instance, problem_statement,0, conversation))
    return data_list


def load_data_aider_bench(file_path):
    data_list = []
    directory_name = os.path.dirname(file_path)
    print('Directory name: ', directory_name)
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            instance_id = data.get('instance_id')
            test_result = data.get('test_result', {})
            resolved = (
                1
                if test_result.get('exit_code') == 0
                and bool(re.fullmatch(r'\.+', test_result.get('test_cases')))
                else 0
            )
            test_cases = test_result.get('test_cases')
            instruction = data.get('instruction')
            agent_trajectory = []
            for step in data.get('history', []):
                if step[0]['source'] != 'agent':
                    continue
                agent_trajectory.append(
                    {
                        'action': step[0].get('action'),
                        'code': step[0].get('args', {}).get('code'),
                        'thought': step[0].get('args', {}).get('thought'),
                        'observation': step[1].get('message'),
                    }
                )
            data_list.append(
                (instance_id, instruction, resolved, test_cases, agent_trajectory)
            )

    return data_list


def get_model_name_aider_bench(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        data = json.loads(first_line)
        return (
            data.get('metadata', {}).get('llm_config', {}).get('model').split('/')[-1]
        )
