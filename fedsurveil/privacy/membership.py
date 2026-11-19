"""Loss-threshold membership inference (Yeom et al. 2018): a simple measure of privacy leakage.

The attacker guesses that an image was in the training set if the model's loss on it is low.
AUROC 0.5 means the attack does no better than chance (no leakage detected by this test); values
well above 0.5 mean the model treats training images differently from new ones. This is a
measurement of our own model on public benchmark data, not an attack on real people.
"""

import numpy as np
import torch
import torch.nn.functional as F

from fedsurveil.fl.client import to_batch


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


@torch.no_grad()
def per_sample_loss(model, x: np.ndarray, y: np.ndarray, device="cpu", batch_size=512):
    model.to(device).eval()
    out = []
    for i in range(0, len(x), batch_size):
        logits = model(to_batch(x[i : i + batch_size], device))
        yb = torch.from_numpy(y[i : i + batch_size]).to(device)
        out.append(F.cross_entropy(logits, yb, reduction="none").cpu().numpy())
    return np.concatenate(out)


def membership_auroc(model, train_xy, test_xy, n=1000, seed=0, device="cpu") -> float:
    """AUROC of "low loss means member" using up to n training and n unseen test images."""
    rng = np.random.default_rng(seed)
    (xt, yt), (xe, ye) = train_xy, test_xy
    ti = rng.choice(len(xt), size=min(n, len(xt)), replace=False)
    ei = rng.choice(len(xe), size=min(n, len(xe)), replace=False)
    member = -per_sample_loss(model, xt[ti], yt[ti], device)
    non_member = -per_sample_loss(model, xe[ei], ye[ei], device)
    return auroc(member, non_member)
