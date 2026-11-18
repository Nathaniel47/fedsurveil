"""One site's local training, and shared helpers to turn uint8 images into model input."""

import numpy as np
import torch
from torch import nn


def to_batch(x: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """uint8 N x H x W x C -> float N x C x H x W scaled to [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).float()
    return (t / 255.0 - 0.5) / 0.5


def local_train(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
    momentum: float = 0.0,
) -> float:
    """Train `model` in place on one site's data. Returns the mean loss of the last epoch."""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    last = 0.0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        total, count = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = np.sort(order[start : start + batch_size])
            xb = to_batch(x[idx], device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        last = total / max(1, count)
    return last


@torch.no_grad()
def predict(
    model: nn.Module, x: np.ndarray, device: torch.device | str = "cpu", batch_size: int = 512
) -> np.ndarray:
    model.to(device).eval()
    out = [
        model(to_batch(x[i : i + batch_size], device)).argmax(1).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
