from tokenizers import pre_tokenizers
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import requests
import random
from pynvml import *
import json
from glob import glob
import os
import torch
import bitsandbytes as bnb
from peft.tuners.lora import LoraLayer        
from omegaconf import OmegaConf

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


def get_model(name, **kwargs):
    model_config = AutoConfig.from_pretrained(name)
    for mapping in MODEL_MAPPINGS:
        model = mapping.get(type(model_config), None)
        if model is not None:
            return model.from_pretrained(name, config=model_config, 
                                         **kwargs)

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
        
        
def get_lora_modules(model, cfg):
    
    modules = cfg.LoraConfig.target_modules
    cls = bnb.nn.Linear4bit if cfg.load_in_4_bit == 4 else (bnb.nn.Linear8bitLt if cfg.load_in_8_bit == 8 else torch.nn.Linear)
    if modules != "all":
        return modules

    modules = {
        name.split('.')[-1]
        for name, module in model.named_modules()
        if isinstance(module, cls)
    }
    if 'lm_head' in modules:
        modules.remove('lm_head')
    return list(modules)
            
    
def prepare_model_types(model, cfg):
    
    for name, module in model.named_modules():
        if isinstance(model, LoraLayer):
            if cfg.trainer.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if cfg.trainer.bf16:
                    module = module.to(torch.bfloat16)
    return model
            