"""SCAFFOLD control variates (Karimireddy et al. 2020, option II).

Each site keeps a control variate c_i that estimates its own gradient direction, and the server
keeps their average c. During local training every gradient is corrected by (c - c_i), which
cancels the "client drift" caused by non-IID data. After training, the site updates c_i from how
far its model actually moved.
"""

import torch

Controls = list[torch.Tensor]


def zero_controls(model: torch.nn.Module) -> Controls:
    return [torch.zeros_like(p, device="cpu") for p in model.parameters()]


def effective_lr(lr: float, momentum: float) -> float:
    """Step size that matters for the control-variate formula when SGD uses momentum.

    SCAFFOLD is derived for plain SGD. With momentum m a constant gradient moves the weights by
    lr / (1 - m) per step in the long run, so that is the value used. This is an approximation.
    """
    return lr / (1.0 - momentum)


def update_site_control(
    c_site: Controls,
    c_server: Controls,
    global_params: Controls,
    local_params: Controls,
    steps: int,
    lr_eff: float,
) -> tuple[Controls, Controls]:
    """Return (new c_i, delta c_i) with c_i+ = c_i - c + (x_global - x_local) / (steps * lr)."""
    new, delta = [], []
    for ci, c, xg, xl in zip(c_site, c_server, global_params, local_params, strict=True):
        ci_new = ci - c + (xg - xl) / (steps * lr_eff)
        new.append(ci_new)
        delta.append(ci_new - ci)
    return new, delta


def update_server_control(c_server: Controls, deltas: list[Controls], num_sites: int) -> Controls:
    """c += (1 / total sites) * sum of the delta c_i of the sites that took part."""
    return [c + sum(d[k] for d in deltas) / num_sites for k, c in enumerate(c_server)]
