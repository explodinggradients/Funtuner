from peft import PeftModel
from huggingface_hub import hf_hub_download
from funtuner.utils import get_model
from transformers import AutoTokenizer
from funtuner.custom_datasets.sftdataset import PromptFormater
from typing import List, Optional
import json
import torch

class Inference:
    def __init__(
        self,
        model_name:str,
        load_in_8bit:bool=False,
    ):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        funtuner_config = hf_hub_download(repo_id = model_name, filename="funtuner_config.json", local_dir=".")
        funtuner_config = json.load(open("funtuner_config.json"))
        model = get_model(funtuner_config["model"], load_in_8bit)
        self.model = PeftModel.from_pretrained(model, model_name).eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.template = PromptFormater(funtuner_config["template"])
        
    def generate(self,
                 instruction:str,
                 context:Optional[str]=None,
                 **kwargs,
    ):
        
        text = self.template.format(instruction, context)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        kwargs |= {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            output = self.model.generate(**kwargs)
        output = self.tokenizer.decode(output)
        return self.template.response(output)
    
    def batch_generate(
        self,
        inputs: List[List[str]],
        **kwargs,
    ):
        # TODO: Add batch_size and iterate if needed
        format_inputs = [item if (len(item) == 2 and item[-1] != "") else [item[0],None] for item in inputs ]
        format_inputs = [self.template.format(instruction, context) for instruction, context in format_inputs]
        format_inputs = self.tokenizer.batch_encode_plus(format_inputs, return_attention_mask=True,
                                                  return_tensors="pt", padding="longest").to(self.device)
        kwargs |= {
            "input_ids":format_inputs["input_ids"],
            "attention_mask":format_inputs["attention_mask"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        output = self.model.generate(**kwargs)
        output = self.tokenizer.batch_decode(output)
        return [self.template.response(text) for text in output]
        
        
        
        


if __name__ == "__main__":
    
    model = Inference("shahules786/GPTNeo-125M-lora")
    kwargs = {"temperature":0.1,
        "top_p":0.75,
        "top_k":5,
        "num_beams":2,
        "max_new_tokens":128,}
    print(model.generate("Which is a species of fish? Tope or Rope", **kwargs))