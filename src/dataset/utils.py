from collections.abc import Generator
from itertools import repeat

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def inf_loop(dataloader: DataLoader) -> Generator:
    """
    Wrapper function for endless dataloader.

    Used for iteration-based training scheme.

    Parameters
    ----------
    dataloader : DataLoader
        Classic finite dataloader.

    Yields
    ------
    Any
        Batch from the dataloader, cycling infinitely.
    """
    for loader in repeat(dataloader):
        yield from loader


def _to_device(
    batch_transforms: dict[str, dict[str, nn.Sequential]],
    device: str,
) -> None:
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Parameters
    ----------
    batch_transforms : dict[str, dict[str, nn.Sequential]]
        Transforms that should be applied on the whole batch.
        Depend on the tensor name.
    device : str
        Device to use for batch transforms.
    """
    for split, transforms in batch_transforms.items():
        if transforms is not None:
            for tensor_name, transform in transforms.items():
                transforms[tensor_name] = transform.to(device)


def get_dataloaders(
    cfg: DictConfig, device: str
) -> tuple[
    dict[str, DataLoader],
    dict[str, dict[str, nn.Sequential]],
]:
    """
    Create dataloaders for each of the dataset partitions.

    Also creates instance and batch transforms.

    Parameters
    ----------
    cfg : DictConfig
        Hydra experiment config.
    device : str
        Device to use for batch transforms.

    Returns
    -------
    dataloaders : dict[str, DataLoader]
        Dict containing dataloader for a partition defined by key.
    batch_transforms : dict[str, dict[str, nn.Sequential]]
        Transforms that should be applied on the whole batch.
        Depend on the tensor name.
    """

    batch_transforms = instantiate(cfg.transforms.batch_transforms, device)
    _to_device(batch_transforms, device)
    datasets = instantiate(cfg.dataset)

    dataloaders = {}
    for dataset_partition in cfg.dataset.keys():
        if cfg.dataloader.get(dataset_partition) is None:
            continue

        dataset = datasets[dataset_partition]

        assert cfg.dataloader[dataset_partition].batch_size <= len(dataset), (
            f'The batch size ({cfg.dataloader[dataset_partition].batch_size}) cannot '
            f'be larger than the dataset length ({len(dataset)})'
        )

        partition_dataloader = instantiate(cfg.dataloader[dataset_partition], dataset=dataset)
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
