from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Union, Optional
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass
from datasets import Split
import json
from omegaconf import OmegaConf
import torch


class PromptFormater:
    def __init__(self, template):
        self.template = json.load(open("funtuner/config/templates.json"))[template]

    def format(
        self,
        instruction: str,
        context: Optional[str] = None,
    ):
        return (
            self.template["prompt_and_input"].format(
                instruction=instruction, context=context
            )
            if context is not None
            else self.template["prompt_only"].format(instruction=instruction)
        )


class FunDataset(Dataset):
    def __init__(
        self,
        name: str = "databricks/databricks-dolly-15k",
        split: Optional[Union[str, Split]] = "train",
        template: str = "alpaca-lora",
        **kwargs,
    ):
        split = OmegaConf.to_object(split)
        if isinstance(split, list) and len(split) == 1:
            split = split[0]
        self.dataset = load_dataset(name, split=split)
        self.prompt = kwargs.get("prompt", "instruction")
        self.context = kwargs.get("context", None)
        self.response = kwargs.get("response", "response")
        for col in [self.prompt, self.context, self.response]:
            if (col is not None) and (col not in self.dataset.features.keys()):
                raise ValueError(f"feature {col} is not present in {name}")

        self.prompt_formater = PromptFormater(template)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        prompt, response = item[self.prompt], item[self.response]
        if self.context is not None:
            context = item[self.context]
        else:
            context = None

        return self.prompt_formater.format(prompt, context), response


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
            input_ids = prompt + rsp
            input_len = len(input_ids)
            if input_len > (self.max_length - 1):
                trun_len = (input_len - self.max_length + 1)
                input_ids = input_ids[:-trun_len]
                    
            input_ids += [self.tokenizer.eos_token_id]
            label_ids = input_ids.copy()
            label_ids[: len(prompt)] = [-100] * min(len(input_ids), len(prompt))
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
        batch_label_ids = torch.where(batch_label_ids==self.tokenizer.pad_token_id, -100, batch_label_ids)

        batch["labels"] = batch_label_ids
        return batch
