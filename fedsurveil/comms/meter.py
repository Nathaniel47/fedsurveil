"""Count bytes moved per round, so communication cost is measured, not estimated."""

import torch


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


def round_bytes(model_bytes: int, num_sites: int, payload_factor: int = 1) -> dict[str, int]:
    """Each selected site downloads the global model and uploads one full update.

    payload_factor is 2 for SCAFFOLD, which also sends the server control variate down and its
    own control-variate change up, each the size of the model.
    """
    total = model_bytes * num_sites * payload_factor
    return {"bytes_down": total, "bytes_up": total}


def secagg_bytes(num_sites: int, key_bytes: int = 32) -> int:
    """Extra upload for the simulated secure aggregation: one shared secret per pair of sites.

    Each site sends key_bytes to every other site (relayed by the server). The masked update
    itself is the same size as a plain one (a 32-bit integer per weight).
    """
    return num_sites * (num_sites - 1) * key_bytes
