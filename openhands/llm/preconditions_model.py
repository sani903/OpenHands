# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import sys

from openai import OpenAI


class LocalPreConditionsModel:
    def __init__(self, model_path):
        # dry run without generations
        if not model_path or model_path == 'test':
            self.model = None
        elif (
            model_path.startswith('openai')
            or model_path.startswith('neulab')
            or model_path.startswith('litellm')
        ):
            self.model = model_path
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         model_path, torch_dtype=torch.bfloat16, device_map='auto'
        #     )

    async def generate_preconditions(self, prompt):
        if not self.model:
            checklist = 'Sample checklist'
        elif isinstance(self.model, str) and (
            self.model.startswith('openai')
            or self.model.startswith('neulab')
            or self.model.startswith('litellm')
        ):
            checklist_generation_prompt = (
                'You are analyzing a software engineering task.\n'
                'Your task is to generate a checklist of key information a developer would need to fully complete the task based only on its <issue_description>.\n'
                '1. Identify the most critical pieces of information required to understand and complete the requirements in <issue_description>.'
                '2. Ensure the checklist is concise and limited to a maximum of 5 essential items.'
                f'<issue_description>\n'
                f'{prompt}\n'
                '</issue_description>\n\n'
                'Format each checklist item within <checklist_item> and </checklist_item> tags.'
            )
            client = OpenAI(
                api_key="sk-baRON8zoJp23Pg9j_6ld3Q",
                base_url='https://cmu.litellm.ai',
            )
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': checklist_generation_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                )
                response_statement = response.choices[0].message.content
                try:
                    checklist = ''
                    items = re.findall(
                        r'<checklist_item>(.*?)</checklist_item>',
                        response_statement,
                        re.DOTALL,
                    )
                    # Convert items to a bullet point list format
                    if items:
                        formatted_list = '\n'.join(
                            [f'- {item.strip()}' for item in items]
                        )
                        checklist = formatted_list
                except Exception as e:
                    print(f'Error calling model API: {e}')
                    if "authentication" in str(e).lower() or "api key" in str(e).lower():
                        checklist = "Authentication error with LLM API"
                    elif "rate limit" in str(e).lower():
                        checklist = "Rate limit exceeded with LLM API"
                    else:
                        checklist = "Error generating checklist"
                    return checklist
            except Exception as e:
                checklist = ''
                print(f'Error calling model API: {e}')

        # else:
        #     checklist_generation_prompt = (
        #         'You are analyzing a software engineering task.\n'
        #         'Your task is to generate a checklist of key information a developer would need to fully complete the task based only on its <issue_description>.\n'
        #         '1. Identify the most critical pieces of information required to understand and complete the requirements in <issue_description>.'
        #         '2. Ensure the checklist is concise and limited to a maximum of 5 essential items.'
        #         f'<issue_description>\n'
        #         f'{prompt}\n'
        #         '</issue_description>\n\n'
        #     )

        #     inputs = self.tokenizer(
        #         checklist_generation_prompt, return_tensors='pt'
        #     ).to(self.model.device)
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=256,
        #         temperature=0.7,
        #         top_p=0.95,
        #         do_sample=True,
        #     )

        #     checklist = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return checklist
