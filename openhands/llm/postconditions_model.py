import re
import json
from openai import OpenAI
from openhands.core.logger import openhands_logger as logger

class LocalPostConditionsModel:
    def __init__(self, model_path):
        # df = pd.read_csv("swe_bench_checklists.csv")
        # self.checklist = df.loc[df['instance_id'] == model_path, 'generated_checklist'].values[0]

        self.model_path = model_path
        self.model = None
        self.test_data = {}

        if not model_path or model_path == 'test':
            if model_path == 'test':
                with open("oracle_tests.json", "r") as f:
                    self.test_data = json.load(f)
        elif model_path.startswith(('openai', 'neulab', 'litellm')):
            self.model = model_path
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map='auto'
            )

    async def generate_postconditions(self, prompt, trajectory=None):
        # When in test mode, prompt is treated as instance_id
        if self.model_path == 'test':
            entry = self.test_data.get(prompt, {})
            # checklist = entry.get("checklist", "No checklist available").strip()
            test_file = entry.get("test_file", "No test file available").strip()
            return (
                # f"\n{checklist}\n\n"
                f"TESTS:\n{test_file}\n"
            )

        # For API models
        # if isinstance(self.model, str):
        #     client = OpenAI(
        #         api_key='sk-baRON8zoJp23Pg9j_6ld3Q',
        #         base_url='https://cmu.litellm.ai'
        #     )

        #     checklist_prompt = (
        #         'You are analyzing a software engineering task.\n'
        #         'Generate a checklist of criteria that must be satisfied for a solution to be successful, '
        #         'based on the <issue_description>. Do not simply say that a test case must be passed. '
        #         'The checklist will be used to verify the solution from an agent trajectory and must be very thorough.\n'
        #         'A few checklist items should also evaluate the trajectory used to arrive at the solution.\n'
        #         f'<issue_description>\n{prompt}\n</issue_description>\n\n'
        #         'Wrap each checklist item with <checklist_item>...</checklist_item>.'
        #     )

        #     response = client.chat.completions.create(
        #         model=self.model,
        #         messages=[
        #             {'role': 'system', 'content': 'You are a helpful assistant.'},
        #             {'role': 'user', 'content': checklist_prompt},
        #         ],
        #         temperature=0.0,
        #         max_tokens=2000,
        #     )

        #     raw = response.choices[0].message.content
        #     items = re.findall(r'<checklist_item>(.*?)</checklist_item>', raw, re.DOTALL)
        #     return '\n'.join(f'- {item.strip()}' for item in items) if items else raw

        # # Local model
        # inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        # output = self.model.generate(
        #     **inputs,
        #     max_new_tokens=256,
        #     temperature=0.7,
        #     top_p=0.95,
        #     do_sample=True,
        # )
        # return self.tokenizer.decode(output[0], skip_special_tokens=True)
