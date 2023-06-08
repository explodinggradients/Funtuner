from tokenizers import pre_tokenizers
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
)

MODEL_MAPPINGS = [MODEL_FOR_CAUSAL_LM_MAPPING]


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

    return tokenizer


def get_model(name):
    model_config = AutoConfig.from_pretrained(name)
    for mapping in MODEL_MAPPINGS:
        model = mapping.get(type(model_config), None)
        if model is not None:
            return model.from_pretrained(name, config=model_config)
