"""Shared pieces of the Flower demo: which data a site holds, and model <-> Flower array records.

Every site works out its own share of the data from (partition id, number of sites, alpha, seed),
so the server and the SuperNodes never have to send data to each other, only model weights.
"""

import numpy as np
import torch

from fedsurveil.data.load import ImageSet
from fedsurveil.data.partition import dirichlet_partition
from fedsurveil.fl.simulate import load_dataset
from fedsurveil.models.cnn import build_model, default_model_name


def limited(data: ImageSet, limit: int | None, seed: int) -> ImageSet:
    """A seeded random subset of the training set (the demo is meant to run on a laptop)."""
    if not limit or limit >= len(data.x_train):
        return data
    keep = np.sort(np.random.default_rng(seed).choice(len(data.x_train), limit, replace=False))
    return ImageSet(
        data.name, data.x_train[keep], data.y_train[keep], data.x_val, data.y_val,
        data.x_test, data.y_test,
    )  # fmt: skip


def site_indices(
    labels: np.ndarray, num_sites: int, alpha: float, seed: int, min_size: int = 10
) -> list[np.ndarray]:
    return dirichlet_partition(labels, num_sites, alpha, seed, min_size=min_size)


def load_site(
    dataset: str, partition_id: int, num_sites: int, alpha: float, seed: int, limit: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Training images and labels held by one site."""
    data = limited(load_dataset(dataset), limit, seed)
    idx = site_indices(data.y_train, num_sites, alpha, seed)[partition_id]
    return data.x_train[idx], data.y_train[idx]


def new_model(dataset: str, seed: int) -> torch.nn.Module:
    """Same architecture and seeded initial weights on the server and on every site."""
    data = load_dataset(dataset)
    torch.manual_seed(seed)
    return build_model(default_model_name(dataset), data.x_train.shape[-1], data.num_classes)
