"""Split a training set across simulated sites with a Dirichlet label skew and uneven sizes.

Small alpha means each site sees only a few classes (strongly non-IID). Large alpha means every
site looks like the whole dataset. Only the training set is partitioned: the global validation
and test sets stay whole and balanced across the real class mix.
"""

import hashlib
import json

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_sites: int,
    alpha: float,
    seed: int,
    size_sigma: float = 0.5,
    min_size: int = 10,
    max_tries: int = 200,
) -> list[np.ndarray]:
    """Return one sorted index array per site. Deterministic for a given seed.

    For each class, its images are shared among sites with proportions drawn from
    Dirichlet(alpha), multiplied by a per-site weight (lognormal, `size_sigma`) so site sizes
    differ even when alpha is large. Redrawn until every site has at least `min_size` images.
    """
    labels = np.asarray(labels)
    if num_sites < 1 or alpha <= 0:
        raise ValueError("num_sites must be >= 1 and alpha > 0")
    if len(labels) < num_sites * min_size:
        raise ValueError("not enough samples for this many sites")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    site_weight = rng.lognormal(mean=0.0, sigma=size_sigma, size=num_sites)

    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(num_sites)]
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(num_sites, alpha)) * site_weight
            p = p / p.sum()
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for site, chunk in enumerate(np.split(idx, cuts)):
                parts[site].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p), dtype=np.int64) for p in parts]
    raise RuntimeError("could not satisfy min_size; raise alpha or lower num_sites/min_size")


def class_matrix(labels: np.ndarray, parts: list[np.ndarray], num_classes: int) -> np.ndarray:
    """Sites x classes table of image counts."""
    m = np.zeros((len(parts), num_classes), dtype=np.int64)
    for i, idx in enumerate(parts):
        m[i] = np.bincount(labels[idx], minlength=num_classes)
    return m


def partition_digest(parts: list[np.ndarray]) -> str:
    """Short hash of a partition, so a saved split can be checked without committing indices."""
    h = hashlib.sha256()
    for p in parts:
        h.update(np.asarray(p, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


def save_partition(path, parts: list[np.ndarray], meta: dict) -> None:
    sites = [p.tolist() for p in parts]
    payload = {"meta": meta, "digest": partition_digest(parts), "sites": sites}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_partition(path) -> list[np.ndarray]:
    payload = json.loads(path.read_text())
    parts = [np.array(p, dtype=np.int64) for p in payload["sites"]]
    if partition_digest(parts) != payload["digest"]:
        raise ValueError(f"{path} is corrupted: digest mismatch")
    return parts
