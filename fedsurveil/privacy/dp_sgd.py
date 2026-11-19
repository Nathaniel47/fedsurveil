"""Sample-level differential privacy inside each site: DP-SGD with Opacus (Abadi et al. 2016).

Guarantee: it hides whether any ONE IMAGE of a site was in that site's training set, for anyone
who sees that site's updates. It does not hide that the site took part (that is client-level DP,
client_dp.py). Each local step clips every image's gradient to norm <= max_grad_norm and adds
Gaussian noise with standard deviation noise_multiplier * max_grad_norm.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedsurveil.fl.client import to_batch


def local_train_dp(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    noise_multiplier: float,
    max_grad_norm: float,
    device: torch.device | str = "cpu",
    momentum: float = 0.0,
) -> list[tuple[float, float, int]]:
    """Train `model` in place with DP-SGD. Returns the accountant history for this call:
    a list of (noise_multiplier, sample_rate, steps) to feed to the RDP accountant."""
    from opacus import PrivacyEngine

    model.to(device).train()
    dataset = TensorDataset(to_batch(x, "cpu"), torch.from_numpy(y))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    engine = PrivacyEngine(accountant="rdp")
    wrapped, dp_opt, dp_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        noise_generator=torch.Generator(device=device).manual_seed(seed),
    )
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in dp_loader:
            if len(xb) == 0:
                continue
            dp_opt.zero_grad()
            loss_fn(wrapped(xb.to(device)), yb.to(device)).backward()
            dp_opt.step()
    return list(engine.accountant.history)
