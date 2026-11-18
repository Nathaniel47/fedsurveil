"""Build a small IP102 subset (top-K classes, 64 px) from the extracted official download.

IP102 is a 3 GB Google Drive folder and "free for academic usage" only, so it is downloaded by
hand and never redistributed. Download `ip102_v1.1` from
https://github.com/xpwu95/IP102 (Drive link in the README), extract it, then run:

    python -m fedsurveil.data.ip102 --root <folder with images/ and train.txt val.txt test.txt>

The split files have one line per image: `<file name> <class id>`.
"""

import argparse
from pathlib import Path

import numpy as np

from fedsurveil import config


def read_split(root: Path, split: str) -> list[tuple[str, int]]:
    rows = []
    for line in (root / f"{split}.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append((parts[0], int(parts[1])))
    return rows


def top_classes(train_rows: list[tuple[str, int]], k: int) -> list[int]:
    """The k classes with most training images (ties broken by class id, so it is deterministic)."""
    counts: dict[int, int] = {}
    for _, label in train_rows:
        counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts, key=lambda c: (-counts[c], c))
    return sorted(ranked[:k])


def _load_image(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def build_subset(root: Path, k: int, size: int) -> dict[str, np.ndarray]:
    splits = {s: read_split(root, s) for s in ("train", "val", "test")}
    keep = top_classes(splits["train"], k)
    remap = {c: i for i, c in enumerate(keep)}
    out: dict[str, np.ndarray] = {"original_class_ids": np.array(keep)}
    for split, rows in splits.items():
        rows = [(f, remap[c]) for f, c in rows if c in remap]
        images, labels = [], []
        for name, label in rows:
            path = root / "images" / name
            if not path.exists():
                continue
            images.append(_load_image(path, size))
            labels.append(label)
        out[f"x_{split}"] = np.stack(images) if images else np.zeros((0, size, size, 3), np.uint8)
        out[f"y_{split}"] = np.array(labels, dtype=np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=config.IP102_NUM_CLASSES)
    parser.add_argument("--size", type=int, default=config.IP102_IMAGE_SIZE)
    args = parser.parse_args()
    data = build_subset(args.root, args.classes, args.size)
    target = config.DATA_DIR / "raw"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "ip102_subset.npz", **data)
    print({k: v.shape for k, v in data.items()})


if __name__ == "__main__":
    main()
