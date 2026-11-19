"""Client-level differential privacy for federated averaging (DP-FedAvg, McMahan et al. 2018).

Guarantee: it hides whether any ONE SITE took part in training at all (its whole dataset), as seen
by anyone who only gets the final model. This is a different, stronger-per-site guarantee than
sample-level DP-SGD, which hides one image. See dp_sgd.py.

Mechanism: every site's update is clipped to L2 norm <= C, the server averages them with equal
weights and adds Gaussian noise with standard deviation (noise_multiplier * C / n_sites).
"""

import math

import torch

State = dict[str, torch.Tensor]

# Orders for the Renyi-DP conversion. Same idea as Opacus' default grid.
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def flatten(state: State) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in state.values()])


def unflatten(vector: torch.Tensor, like: State) -> State:
    out, offset = {}, 0
    for key, ref in like.items():
        n = ref.numel()
        out[key] = vector[offset : offset + n].reshape(ref.shape).to(ref.dtype)
        offset += n
    return out


def clip_vector(vector: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, float, bool]:
    """Scale the vector to L2 norm <= max_norm. Returns (clipped, original norm, was_clipped)."""
    norm = float(vector.norm())
    if norm <= max_norm:
        return vector, norm, False
    return vector * (max_norm / norm), norm, True


def gaussian_rdp_epsilon(noise_multiplier: float, steps: int, delta: float) -> float:
    """(epsilon, delta)-DP of `steps` full-participation Gaussian mechanisms.

    Every site is used every round (no subsampling), so the Renyi-DP of order a is
    steps * a / (2 sigma^2). Converted to (epsilon, delta) with the standard tight conversion.
    """
    if noise_multiplier <= 0:
        return math.inf
    best = math.inf
    for a in ORDERS:
        rdp = steps * a / (2 * noise_multiplier**2)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return best


def sampled_epsilon(history: list[tuple[float, float, int]], delta: float) -> float:
    """Epsilon for a list of (noise_multiplier, sample_rate, steps) using Opacus' RDP accountant.

    The accountant assumes each record is included independently with probability sample_rate
    (Poisson sampling). Picking a fixed number of sites per round is close to that but not
    identical, so for partial participation the result is an approximation.
    """
    from opacus.accountants import RDPAccountant

    accountant = RDPAccountant()
    accountant.history = list(history)
    return float(accountant.get_epsilon(delta))


def client_level_epsilon(
    noise_multiplier: float, sites_per_round: int, num_sites: int, rounds: int, delta: float
) -> float:
    if noise_multiplier <= 0:
        return math.inf
    if sites_per_round >= num_sites:
        return gaussian_rdp_epsilon(noise_multiplier, rounds, delta)
    return sampled_epsilon([(noise_multiplier, sites_per_round / num_sites, rounds)], delta)
