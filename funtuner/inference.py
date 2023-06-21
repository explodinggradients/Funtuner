from peft import PeftModel
from huggingface_hub import hf_hub_download
from funtuner.utils import get_model
from transformers import AutoTokenizer
from funtuner.custom_datasets.sftdataset import PromptFormater
 

class Inference:
    def __init__(
        self,
        model_name:str,
        load_in_8bit:bool=False,
    ):
        
        funtuner_config = hf_hub_download(repo_id = model_name, filename="funtuner_config.json")
        model = get_model(funtuner_config["model"], load_in_8bit)
        self.model = PeftModel.from_pretrained(model, model_name).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.template = PromptFormater(funtuner_config["template"])
        
    def generate(self,
                 instruction:str,
                 input:Optional[str],
                 **kwargs,
    ):
        
        input_ids = self.tokenizer(text, return_tensors="pt")
        output = self.model.generate(input_ids,
                            **kwargs)
        output = output.sequences[0]
        output = tokenizer.decode(output)
        return output