"""FedAvg aggregation: the new global model is the size-weighted average of the site models."""

import torch

State = dict[str, torch.Tensor]


def state_delta(new: State, old: State) -> State:
    return {k: new[k].detach().cpu() - old[k].detach().cpu() for k in old}


def fedavg_aggregate(global_state: State, deltas: list[State], sizes: list[int]) -> State:
    """global + sum_i (n_i / N) * delta_i, with N the total number of images in the round."""
    if not deltas or len(deltas) != len(sizes):
        raise ValueError("need one size per update, and at least one update")
    total = float(sum(sizes))
    out = {}
    for key, value in global_state.items():
        step = sum(d[key] * (n / total) for d, n in zip(deltas, sizes, strict=True))
        out[key] = value.detach().cpu() + step
    return out
