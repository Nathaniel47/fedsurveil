"""Lossy compression of a site's model update, to save upload bandwidth.

Two simple schemes, both applied to the flattened update (the change in weights over a round):
- int8: every weight is stored as one signed byte plus one shared float32 scale (about 4x smaller).
- top-k: only the k largest-magnitude weights are sent, each as a 4-byte value and a 4-byte
  position (so keeping 10% costs about 20% of the dense size).

Top-k throws most of the update away, which hurts accuracy unless the discarded part is remembered
and added to the next round's update ("error feedback", Stich et al. 2018). The compressor keeps
that memory per site.
"""

import torch

FLOAT_BYTES = 4
INDEX_BYTES = 4
KINDS = ("none", "int8", "topk")


def quantize_int8(vec: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return (decoded vector, bytes sent)."""
    peak = float(vec.abs().max())
    scale = peak / 127.0 if peak > 0 else 1.0
    q = torch.clamp(torch.round(vec / scale), -127, 127).to(torch.int8)
    return q.float() * scale, vec.numel() + FLOAT_BYTES


def sparsify_topk(vec: torch.Tensor, fraction: float) -> tuple[torch.Tensor, int]:
    """Keep the largest-magnitude `fraction` of the entries. Return (decoded, bytes sent)."""
    k = max(1, int(round(vec.numel() * fraction)))
    idx = torch.topk(vec.abs(), k).indices
    out = torch.zeros_like(vec)
    out[idx] = vec[idx]
    return out, k * (FLOAT_BYTES + INDEX_BYTES)


def dense_bytes(num_values: int) -> int:
    return num_values * FLOAT_BYTES


class UpdateCompressor:
    """Per-site compression with error feedback for top-k."""

    def __init__(self, kind: str, num_sites: int, topk_fraction: float = 0.1):
        if kind not in KINDS:
            raise ValueError(f"unknown compression {kind!r}: choose from {KINDS}")
        if kind == "topk" and not 0 < topk_fraction <= 1:
            raise ValueError("topk_fraction must be in (0, 1]")
        self.kind = kind
        self.fraction = topk_fraction
        self.residual: dict[int, torch.Tensor] = {}
        self.num_sites = num_sites

    def __call__(self, site: int, vec: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self.kind == "none":
            return vec, dense_bytes(vec.numel())
        if self.kind == "int8":
            return quantize_int8(vec)
        carried = vec + self.residual.get(site, torch.zeros_like(vec))
        decoded, nbytes = sparsify_topk(carried, self.fraction)
        self.residual[site] = carried - decoded
        return decoded, nbytes
