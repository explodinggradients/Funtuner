import torch
from torch.utils.data import random_split, ConcatDataset
from funtuner.custom_datasets.sftdataset import FunDataset
import yaml
from pathlib import Path

generator = torch.Generator().manual_seed(42)
DATASET_MAPPING = yaml.safe_load(Path('funtuner/config/datasets.yaml').read_text())


def get_single_dataset(name, split, sep_token):
    args = DATASET_MAPPING.get(name)
    if args is not None:
        dataset = FunDataset(name=name, split=split, sep_token=sep_token)
    else:
        raise ValueError(f"Invalid dataset name {name}. Add dataset to dataset.yaml")

    return dataset


def get_datasets(config):
    dataset_list = []
    sep_token = config.special_tokens.sep_token
    for dataset in config.datasets:
        name = list(dataset.keys())[0]
        splits = dataset[name].get("split", "train")
        dataset_list.append(
            get_single_dataset(
                name,
                splits,
                sep_token,
            )
        )

    dataset = ConcatDataset(dataset_list)
    train_dataset, valid_dataset = random_split(
        dataset,
        [1 - config.validation_size, config.validation_size],
        generator=generator,
    )
    return train_dataset, valid_dataset