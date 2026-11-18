"""Load MedMNIST and IP102-subset arrays as plain numpy, with no torch needed."""

from dataclasses import dataclass

import numpy as np

from fedsurveil import config


@dataclass
class ImageSet:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def num_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_val.max(), self.y_test.max())) + 1


def _to_hwc(x: np.ndarray) -> np.ndarray:
    """Make images N x H x W x C uint8 (MedMNIST greyscale sets have no channel axis)."""
    if x.ndim == 3:
        x = x[..., None]
    return x.astype(np.uint8, copy=False)


def load_medmnist(name: str) -> ImageSet:
    if name not in config.MEDMNIST_DATASETS:
        raise ValueError(f"{name} is not an allowed dataset: {config.MEDMNIST_DATASETS}")
    path = config.DATA_DIR / "raw" / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python -m fedsurveil.data.fetch")
    with np.load(path) as d:
        return ImageSet(
            name=name,
            x_train=_to_hwc(d["train_images"]),
            y_train=d["train_labels"].reshape(-1).astype(np.int64),
            x_val=_to_hwc(d["val_images"]),
            y_val=d["val_labels"].reshape(-1).astype(np.int64),
            x_test=_to_hwc(d["test_images"]),
            y_test=d["test_labels"].reshape(-1).astype(np.int64),
        )


def load_ip102_subset(path=None) -> ImageSet:
    path = path or config.ip102_npz_path() or config.DATA_DIR / "raw" / "ip102_subset.npz"
    with np.load(path) as d:
        return ImageSet(
            name="ip102",
            x_train=d["x_train"],
            y_train=d["y_train"].astype(np.int64),
            x_val=d["x_val"],
            y_val=d["y_val"].astype(np.int64),
            x_test=d["x_test"],
            y_test=d["y_test"].astype(np.int64),
        )
