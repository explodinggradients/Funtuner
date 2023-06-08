import torch
from torch.utils.data import random_split, ConcatDataset
from funtuner.datasets.dolly import DollyDataset

generator = torch.Generator().manual_seed(42)


def get_single_dataset(name, **kwargs):
    if name == "dolly":
        dataset = DollyDataset(**kwargs)
    else:
        raise ValueError(f"Invalid dataset name {name}")

    return dataset


def get_datasets(config):
    dataset_list = []
    for dataset in config.datasets:
        name = list(dataset.keys())[0]
        kwargs = dataset[name]
        dataset_list.append(get_single_dataset(name, **kwargs))

    dataset = ConcatDataset(dataset_list)
    train_dataset, valid_dataset = random_split(
        dataset,
        [1 - config.validation_size, config.validation_size],
        generator=generator,
    )
    return train_dataset, valid_dataset
