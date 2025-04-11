import re

from openai import OpenAI

from openhands.core.logger import openhands_logger as logger


class LocalPreConditionsModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

        if not model_path or model_path == 'test':
            pass
        elif model_path.startswith(('openai', 'neulab', 'litellm')):
            self.model = model_path
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map='auto'
            )

    async def generate_preconditions(self, prompt):

        if not self.model:
            return '- Sample checklist item 1\n- Sample checklist item 2\n- Sample checklist item 3'

        if isinstance(self.model, str):  # API model
            client = OpenAI(
                api_key='', base_url='https://cmu.litellm.ai'
            )

            checklist_prompt = (
                'You are analyzing a software engineering task.\n'
                'Generate a checklist of key information a developer would need to fully complete the task '
                'based only on the <issue_description>. The key information must not be steps in the solution. '
                'Instead they are details that should be included in the issue to make it solvable. ' 
                'Phrase each checklist item as a yes or no question of whether the input contains the particular important information.'
                'Consider the various steps in solving the issue, and based on those steps, what information '
                'might be required in the Github issue. Limit to 5 items.\n'
                f'<issue_description>\n{prompt}\n</issue_description>\n\n'
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
