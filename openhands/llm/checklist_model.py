from openhands.llm.base import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalChecklistModel(BaseLLM):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    async def generate_checklist(self, prompt):
        checklist_generation_prompt = (
            'You are analyzing a software engineering task.\n'
            'Your task is to generate a checklist of key information a developer would need to fully complete the task based only on its <issue_description>.\n'
            '1. Identify the most critical pieces of information required to understand and complete the requirements in <issue_description>.'
            '2. Ensure the checklist is concise and limited to a maximum of 5 essential items.'
            f'<issue_description>\n'
            f'{prompt}\n'
            '</issue_description>\n\n'
        )
        
        inputs = self.tokenizer(checklist_generation_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        checklist = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return checklist
