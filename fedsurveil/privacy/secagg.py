"""SIMULATED secure aggregation by pairwise masking (the idea of Bonawitz et al. 2017).

This is a teaching simulation, NOT real cryptography: the pairwise masks come from a seeded
random generator on one machine, there is no key agreement, no authentication, and no handling of
sites that drop out mid-round. What it does show faithfully is the core trick and its cost:
each site sends a masked vector that looks like random numbers on its own, yet the masks cancel
exactly when the server adds all the vectors, so the server learns only the sum.

Arithmetic is on fixed-point integers modulo 2^32, so the cancellation is exact.
"""

import numpy as np
import torch

MODULUS = 1 << 32
SCALE = 1 << 16  # fixed-point resolution: values are rounded to multiples of 1/65536


def encode(vector: torch.Tensor) -> np.ndarray:
    """float vector -> fixed-point integers (as int64, reduced modulo 2^32)."""
    return np.mod(np.rint(vector.double().numpy() * SCALE).astype(np.int64), MODULUS)


def decode(total: np.ndarray) -> torch.Tensor:
    """Fixed-point integers modulo 2^32 -> float vector, mapping the upper half to negatives."""
    signed = np.where(total >= MODULUS // 2, total - MODULUS, total)
    return torch.from_numpy(signed.astype(np.float64) / SCALE).float()


def _pair_mask(seed: int, i: int, j: int, size: int) -> np.ndarray:
    rng = np.random.default_rng([seed, i, j])
    return rng.integers(0, MODULUS, size=size, dtype=np.int64)


def mask_updates(vectors: list[torch.Tensor], seed: int) -> list[np.ndarray]:
    """What each site would send: its fixed-point update plus/minus a mask for every other site.

    Site i adds the mask shared with each later site j and subtracts the mask shared with each
    earlier site, so every mask appears once with each sign across the whole group.
    """
    n = len(vectors)
    size = vectors[0].numel()
    masked = []
    for i, v in enumerate(vectors):
        total = encode(v)
        for j in range(n):
            if j == i:
                continue
            m = _pair_mask(seed, min(i, j), max(i, j), size)
            total = total + m if j > i else total - m
        masked.append(np.mod(total, MODULUS))
    return masked


def secure_sum(vectors: list[torch.Tensor], seed: int) -> torch.Tensor:
    """Sum of the vectors as the server would compute it: from the masked vectors only."""
    masked = mask_updates(vectors, seed)
    return decode(np.mod(np.sum(masked, axis=0), MODULUS))
