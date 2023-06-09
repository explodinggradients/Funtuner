from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass
from funtuner.utils import SPECIAL_TOKENS

def format_output(prompt, context, response):
    if context != "":
        prompt = (
            SPECIAL_TOKENS["prompt"] + prompt + SPECIAL_TOKENS["context"] + context
        )
    else:
        prompt = SPECIAL_TOKENS["prompt"] + prompt

    response = SPECIAL_TOKENS["response"] + response

    return prompt, response


class FunDataset(Dataset):
    def __init__(
        self,
        name: str = "databricks/databricks-dolly-15k",
        split: Union[List, str] = "train",
        sep_token: str = "[SEP]",
        **kwargs,
    ):
        self.dataset = load_dataset(name, split=kwargs.get("split", "train"))
        self.prompt = kwargs.get("prompt", "instruction")
        self.context = kwargs.get("context", None)
        self.response = kwargs.get("response", "response")
        self.sep_token = sep_token
        for col in [self.prompt, self.context, self.response]:
            if (col is not None) and (col not in self.dataset.features.keys()):
                raise ValueError(f"feature {col} is not present in {name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        prompt, response = item[self.prompt], item[self.response]
        if self.context is not None:
            context = item[self.context]
        else:
            context = ""

        return format_output(prompt, context, response)

@dataclass
class FunDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512

    def __call__(self, batch):
        batch_maxlen = 0
        batch_input_ids, batch_label_ids = [], []
        prompts, responses = zip(*batch)
        prompt_tokens = self.tokenizer.batch_encode_plus(
            prompts, return_attention_mask=False
        ).input_ids
        response_tokens = self.tokenizer.batch_encode_plus(
            responses, return_attention_mask=False
        ).input_ids
        for prompt, rsp in zip(prompt_tokens, response_tokens):
            input_len = len(prompt + rsp)
            if input_len > (self.max_length - 1):
                rsp = rsp[: -(input_len - self.max_length + 1)]
            input_ids = prompt + rsp + [self.tokenizer.eos_token_id]
            label_ids = input_ids.copy()
            label_ids[: len(prompt)] = [-100] * len(prompt)
            if len(input_ids) > batch_maxlen:
                batch_maxlen = len(input_ids)

            batch_input_ids.append(input_ids)
            batch_label_ids.append(label_ids)

        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            max_length=batch_maxlen,
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch_label_ids = self.tokenizer.pad(
            {"input_ids": batch_label_ids},
            max_length=batch_maxlen,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]

        batch["labels"] = batch_label_ids
        return batch