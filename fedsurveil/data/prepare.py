"""Create and save the seeded site partitions for every dataset on disk.

Run: python -m fedsurveil.data.prepare
Indices are written to data/splits/ (gitignored, rebuilt from the seed). Only a short hash per
split is committed, in data/split_digests.json, so anyone can check they rebuilt the same split.
"""

import json

from fedsurveil import config
from fedsurveil.data.load import ImageSet, load_ip102_subset, load_medmnist
from fedsurveil.data.partition import dirichlet_partition, partition_digest, save_partition

SPLITS_DIR = config.DATA_DIR / "splits"
DIGESTS = config.DATA_DIR / "split_digests.json"
DEFAULT_SITES = 10


def split_name(dataset: str, num_sites: int, alpha: float, seed: int) -> str:
    return f"{dataset}_sites{num_sites}_alpha{alpha:g}_seed{seed}"


def available_sets() -> list[ImageSet]:
    sets = []
    for name in config.MEDMNIST_DATASETS:
        if (config.DATA_DIR / "raw" / f"{name}.npz").exists():
            sets.append(load_medmnist(name))
    try:
        sets.append(load_ip102_subset())
    except FileNotFoundError:
        print("ip102_subset.npz not found: skipping IP102 (see fedsurveil/data/ip102.py)")
    return sets


def prepare(num_sites: int = DEFAULT_SITES) -> dict[str, str]:
    digests = json.loads(DIGESTS.read_text()) if DIGESTS.exists() else {}
    for data in available_sets():
        for alpha in config.DIRICHLET_ALPHAS:
            for seed in config.SEEDS:
                parts = dirichlet_partition(data.y_train, num_sites, alpha, seed)
                name = split_name(data.name, num_sites, alpha, seed)
                meta = {"dataset": data.name, "sites": num_sites, "alpha": alpha, "seed": seed}
                save_partition(SPLITS_DIR / f"{name}.json", parts, meta)
                digests[name] = partition_digest(parts)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
    return digests


if __name__ == "__main__":
    result = prepare()
    print(f"saved {len(result)} partitions")
