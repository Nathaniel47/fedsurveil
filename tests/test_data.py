import numpy as np
import pytest
from PIL import Image

from fedsurveil.data import ip102
from fedsurveil.data.load import _to_hwc, load_medmnist
from fedsurveil.data.partition import (
    class_matrix,
    dirichlet_partition,
    load_partition,
    partition_digest,
    save_partition,
)


def make_labels(n=2000, k=5, seed=0):
    return np.random.default_rng(seed).integers(0, k, size=n)


def test_partition_is_disjoint_and_covers_everything():
    labels = make_labels()
    parts = dirichlet_partition(labels, num_sites=8, alpha=0.5, seed=1)
    joined = np.concatenate(parts)
    assert len(joined) == len(labels)
    assert len(np.unique(joined)) == len(labels)


def test_partition_is_deterministic_and_seed_dependent():
    labels = make_labels()
    a = dirichlet_partition(labels, 6, 0.5, seed=3)
    b = dirichlet_partition(labels, 6, 0.5, seed=3)
    c = dirichlet_partition(labels, 6, 0.5, seed=4)
    assert partition_digest(a) == partition_digest(b)
    assert partition_digest(a) != partition_digest(c)


def test_small_alpha_is_more_skewed_than_large_alpha():
    labels = make_labels(n=6000)

    def dominant_share(alpha):
        parts = dirichlet_partition(labels, 10, alpha, seed=0)
        m = class_matrix(labels, parts, 5)
        return (m.max(axis=1) / m.sum(axis=1)).mean()

    assert dominant_share(0.1) > dominant_share(100.0) + 0.2


def test_every_site_meets_min_size():
    parts = dirichlet_partition(make_labels(), 10, 0.1, seed=2, min_size=20)
    assert min(len(p) for p in parts) >= 20


def test_site_sizes_differ_even_when_labels_are_near_iid():
    parts = dirichlet_partition(make_labels(n=6000), 10, 100.0, seed=0, size_sigma=0.5)
    sizes = [len(p) for p in parts]
    assert max(sizes) > 1.3 * min(sizes)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        dirichlet_partition(make_labels(), 0, 0.5, 0)
    with pytest.raises(ValueError):
        dirichlet_partition(make_labels(n=20), 10, 0.5, 0, min_size=10)


def test_saved_partition_round_trips_and_detects_corruption(tmp_path):
    parts = dirichlet_partition(make_labels(), 5, 0.5, seed=0)
    path = tmp_path / "p.json"
    save_partition(path, parts, {"seed": 0})
    loaded = load_partition(path)
    assert partition_digest(loaded) == partition_digest(parts)
    path.write_text(path.read_text().replace("[", "[0,", 1))
    with pytest.raises(ValueError):
        load_partition(path)


def test_greyscale_images_get_a_channel_axis():
    assert _to_hwc(np.zeros((4, 28, 28), dtype=np.uint8)).shape == (4, 28, 28, 1)
    assert _to_hwc(np.zeros((4, 28, 28, 3), dtype=np.uint8)).shape == (4, 28, 28, 3)


def test_dermamnist_cannot_be_loaded():
    with pytest.raises(ValueError):
        load_medmnist("dermamnist")


def test_ip102_subset_keeps_top_classes_and_resizes(tmp_path):
    (tmp_path / "images").mkdir()
    layout = {"train": [(0, 4), (1, 3), (2, 1)], "val": [(0, 1), (2, 1)], "test": [(0, 1), (1, 1)]}
    for split, spec in layout.items():
        lines = []
        for label, n in spec:
            for i in range(n):
                name = f"{split}_{label}_{i}.jpg"
                Image.new("RGB", (100, 80), (label * 40, 0, 0)).save(tmp_path / "images" / name)
                lines.append(f"{name} {label}")
        (tmp_path / f"{split}.txt").write_text("\n".join(lines))
    out = ip102.build_subset(tmp_path, k=2, size=16)
    assert out["original_class_ids"].tolist() == [0, 1]
    assert out["x_train"].shape == (7, 16, 16, 3)
    assert set(out["y_train"].tolist()) == {0, 1}
    assert out["x_val"].shape == (1, 16, 16, 3)
    assert out["x_test"].shape == (2, 16, 16, 3)
