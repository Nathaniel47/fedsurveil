"""Metrics on plain numpy arrays, so they run in CI without torch."""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Mean F1 over classes. A class with no true and no predicted samples scores 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def site_eval_indices(
    y_test: np.ndarray, site_class_counts: np.ndarray, n_per_site: int, seed: int
) -> list[np.ndarray]:
    """Per-site evaluation sets drawn from the global test set.

    Each site gets test images with the same class mix as its training data, so "accuracy on
    site k" means accuracy on the kind of images site k actually sees. Sampled with replacement,
    so a small test class can still fill a site that is dominated by it.
    """
    rng = np.random.default_rng(seed)
    y_test = np.asarray(y_test)
    by_class = [np.flatnonzero(y_test == c) for c in range(site_class_counts.shape[1])]
    out = []
    for counts in site_class_counts:
        p = counts / counts.sum()
        drawn = rng.choice(len(p), size=n_per_site, p=p)
        idx = np.empty(n_per_site, dtype=np.int64)
        for c in np.unique(drawn):
            pool = by_class[c]
            slots = drawn == c
            idx[slots] = rng.choice(pool, size=int(slots.sum())) if len(pool) else 0
        out.append(idx)
    return out


def spread(values: list[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "std": float(v.std())}
