"""The update pipeline between the sites and the new global model.

Order: clip (client-level DP) -> weight -> (simulated secure) sum -> noise (client-level DP).
With clipping on, every site gets equal weight 1/n so that one site can change the sum by at most
C/n, which is what the noise is calibrated to. Without it, sites are weighted by their size as in
plain FedAvg.
"""

import torch

from fedsurveil.privacy.client_dp import clip_vector, flatten, unflatten
from fedsurveil.privacy.secagg import secure_sum

State = dict[str, torch.Tensor]


def private_aggregate(
    global_state: State,
    deltas: list[State],
    sizes: list[int],
    *,
    clip: float = 0.0,
    noise_multiplier: float = 0.0,
    secagg: bool = False,
    seed: int = 0,
) -> tuple[State, dict]:
    """Return (new global state, info). clip = 0 disables clipping and noise."""
    n = len(deltas)
    vectors, norms, clipped = [], [], 0
    for d in deltas:
        v = flatten(d)
        if clip > 0:
            v, norm, was = clip_vector(v, clip)
            clipped += int(was)
            norms.append(norm)
        vectors.append(v)
    if clip > 0:
        weights = [1.0 / n] * n
    else:
        total = float(sum(sizes))
        weights = [s / total for s in sizes]
    weighted = [v * w for v, w in zip(vectors, weights, strict=True)]
    summed = secure_sum(weighted, seed) if secagg else torch.stack(weighted).sum(0)
    if clip > 0 and noise_multiplier > 0:
        gen = torch.Generator().manual_seed(seed)
        std = noise_multiplier * clip / n
        summed = summed + torch.randn(summed.shape, generator=gen) * std
    step = unflatten(summed, global_state)
    new_state = {k: global_state[k].detach().cpu() + step[k] for k in global_state}
    info = {
        "clipped_fraction": clipped / n if clip > 0 else 0.0,
        "mean_update_norm": float(sum(norms) / len(norms)) if norms else None,
    }
    return new_state, info
