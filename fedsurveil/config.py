"""Single place for datasets, seeds, privacy settings and paths. PLAN.md decides the values."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

# DermaMNIST is CC-BY-NC-4.0, so it is deliberately not listed (see PLAN.md).
MEDMNIST_DATASETS = ("pathmnist", "bloodmnist", "organamnist")
MEDMNIST_IMAGE_SIZE = 28
IP102_IMAGE_SIZE = 64
IP102_NUM_CLASSES = 10

ALGORITHMS = ("fedavg", "fedprox", "scaffold")
DIRICHLET_ALPHAS = (0.1, 0.5, 100.0)
NUM_SITES = (5, 10, 20)

DP_DELTA = 1e-5

SEEDS = (0, 1, 2)


def ip102_npz_path() -> Path | None:
    """Location of the preprocessed IP102 subset, from the environment only."""
    value = os.environ.get("IP102_NPZ_PATH")
    return Path(value) if value else None
