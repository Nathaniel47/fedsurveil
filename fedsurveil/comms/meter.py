"""Count bytes moved per round, so communication cost is measured, not estimated."""

import torch


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


def round_bytes(model_bytes: int, num_sites: int) -> dict[str, int]:
    """Each selected site downloads the global model and uploads one full update."""
    return {"bytes_down": model_bytes * num_sites, "bytes_up": model_bytes * num_sites}
