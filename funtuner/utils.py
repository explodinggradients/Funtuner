from tokenizers import pre_tokenizers
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import requests
import random
from pynvml import *
import json
from glob import glob
import os
from transformers import GPTNeoXForCausalLM

MODEL_MAPPINGS = [MODEL_FOR_CAUSAL_LM_MAPPING]


def get_tokenizer(config):
    
    if "llama" not in config.model: 
        tokenizer = AutoTokenizer.from_pretrained(config.model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(config.model)


    if hasattr(config, "per_digit_tokens") and config.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if config.special_tokens:
        special_tokens = {
            "pad_token": config.special_tokens.pad_token,
            "eos_token": config.special_tokens.eos_token,
            "sep_token": config.special_tokens.sep_token,
        }
        tokenizer.add_special_tokens(special_tokens)

    return tokenizer


def get_model(name, load_in_8bit=False):
    model_config = AutoConfig.from_pretrained(name)
    for mapping in MODEL_MAPPINGS:
        model = mapping.get(type(model_config), None)
        if model is not None:
            return model.from_pretrained(name, config=model_config, 
                                         load_in_8bit=load_in_8bit)

def get_name():
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    response = requests.get(word_site)
    WORDS = response.content.splitlines()
    return random.choice(WORDS).decode('UTF-8')


def print_gpu_utilization():    
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")


def save_json(filename, data):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
        
        
def add_additional_config(cfg):
    config_files = glob(os.path.join(cfg.log_dir, "**/*.json"), recursive=True)
    for file in config_files:
        config = json.load(open(file))
        config["template"] = cfg.template
        config["train_max_len"] = cfg.max_length
        save_json(file, config)