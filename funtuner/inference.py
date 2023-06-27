from peft import PeftModel
from funtuner.utils import get_model
from transformers import AutoTokenizer, LlamaTokenizer
from funtuner.custom_datasets.sftdataset import PromptFormater
from typing import List, Optional
import torch
from huggingface_hub import hf_hub_download
import json
import os

class Inference:
    def __init__(
        self,
        model_name:str,
        load_in_8bit:bool=False,
    ):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config = self.load_config(model_name)
        base_model = config["base_model_name_or_path"]

        self.tokenizer = self.load_tokenizer(model_name, base_model)
        model = get_model(base_model, load_in_8bit)
        
        model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(model, model_name).eval()
        if not load_in_8bit:
            self.model = self.model.half()
        self.model.to(self.device)
        self.tokenizer.padding_side = "left"
        self.template = PromptFormater(config.get("template", "alpaca-lora"))
        
    def load_config(self, model_name):
        
        if os.path.exists(model_name):
            config = os.path.join(model_name, "adapter_config.json")
        else:
            config = hf_hub_download(repo_id=model_name, filename="adapter_config.json", local_dir=".")
            config = "adapter_config.json"
            
        config = json.load(open(config))
        return config
    
    def load_tokenizer(self, model_name, base_model):
        if "llama" not in base_model: 
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        return tokenizer
    def generate(self,
                 instruction:str,
                 context:Optional[str]=None,
                 **kwargs,
    ):
        
        text = self.template.format(instruction, context)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        kwargs.update({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        })
        with torch.no_grad():
            output = self.model.generate(**kwargs)[0]
        output = self.tokenizer.decode(output, skip_special_tokens=True)
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
        kwargs.update({
            "input_ids":format_inputs["input_ids"],
            "attention_mask":format_inputs["attention_mask"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        })
        
        output = self.model.generate(**kwargs)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return [self.template.response(text) for text in output]
        
        
        
        


if __name__ == "__main__":
    
    model = Inference("shahules786/GPTNeo-125M-lora")
    kwargs = {"temperature":0.1,
        "top_p":0.75,
        "top_k":5,
        "num_beams":2,
        "max_new_tokens":128,}
    print(model.generate("Which is a species of fish? Tope or Rope", **kwargs))