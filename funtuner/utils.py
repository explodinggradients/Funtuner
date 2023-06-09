from tokenizers import pre_tokenizers
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import requests
import random

MODEL_MAPPINGS = [MODEL_FOR_CAUSAL_LM_MAPPING]
SPECIAL_TOKENS = {
    "prompt": "<|prompt|>",
    "response": "<|response|>",
    "context": "<|context|>",
}

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    if hasattr(config, "per_digit_tokens") and config.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if config.special_tokens:
        special_tokens = {
            "pad_token": config.special_tokens.pad_token,
            "eos_token": config.special_tokens.eos_token,
            "sep_token": config.special_tokens.sep_token,
        }
        tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.values())})

    return tokenizer


def get_model(name):
    model_config = AutoConfig.from_pretrained(name)
    for mapping in MODEL_MAPPINGS:
        model = mapping.get(type(model_config), None)
        if model is not None:
            return model.from_pretrained(name, config=model_config)

def get_name():
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    response = requests.get(word_site)
    WORDS = response.content.splitlines()
    return random.choice(WORDS).decode('UTF-8')