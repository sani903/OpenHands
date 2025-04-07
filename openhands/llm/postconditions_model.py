import re

from openai import OpenAI

from openhands.core.logger import openhands_logger as logger

logger.info('message')


class LocalPostConditionsModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        print(f'[PostModel] Initializing with: {model_path}')

        if not model_path or model_path == 'test':
            print('[PostModel] Using dummy mode')
        elif model_path.startswith(('openai', 'neulab', 'litellm')):
            print(f'[PostModel] Using API model: {model_path}')
            self.model = model_path
        else:
            print(f'[PostModel] Using local model: {model_path}')
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map='auto'
            )

    async def generate_postconditions(self, prompt, trajectory=None):
        print('[PostModel] Generating checklist...')
        trajectory_text = trajectory or 'No trajectory available'

        if not self.model:
            return '- Sample postcondition 1\n- Sample postcondition 2\n- Sample postcondition 3'

        if isinstance(self.model, str):  # API model
            client = OpenAI(
                api_key='sk-baRON8zoJp23Pg9j_6ld3Q', base_url='https://cmu.litellm.ai'
            )

            checklist_prompt = (
                'You are analyzing a software engineering task.\n'
                'Generate a checklist of criteria that must be satisfied for a solution to be successful, '
                'based on the <issue_description> and the <trajectory>. Do not simply say that a test case must be passed. The checklist will be used to verify the solution.\n'
                'Limit to 5 concise items.\n'
                f'<issue_description>\n{prompt}\n</issue_description>\n\n'
                f'<trajectory>\n{trajectory_text}\n</trajectory>\n\n'
                'Wrap each checklist item with <checklist_item>...</checklist_item>.'
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': checklist_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )

            raw = response.choices[0].message.content
            items = re.findall(
                r'<checklist_item>(.*?)</checklist_item>', raw, re.DOTALL
            )
            return '\n'.join(f'- {item.strip()}' for item in items) if items else raw

        else:  # Local model
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
