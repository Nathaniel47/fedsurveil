import numpy as np
import pytest

torch = pytest.importorskip("torch")
flwr_app = pytest.importorskip("flwr.app")

from fedsurveil import config  # noqa: E402
from fedsurveil.data.load import ImageSet  # noqa: E402
from fedsurveil.flower_demo.common import limited, load_site, site_indices  # noqa: E402
from fedsurveil.models.cnn import SmallCNN  # noqa: E402


def labels(n=600, classes=4, seed=0):
    return np.random.default_rng(seed).integers(0, classes, n)


def test_sites_are_disjoint_and_repeatable():
    y = labels()
    a = site_indices(y, 3, alpha=0.5, seed=1)
    b = site_indices(y, 3, alpha=0.5, seed=1)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    seen = np.concatenate(a)
    assert len(seen) == len(set(seen.tolist()))  # no image belongs to two sites


def test_limited_is_a_seeded_subset():
    n = 200
    x = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    data = ImageSet("t", x, labels(n), x[:10], labels(10), x[:10], labels(10))
    a, b = limited(data, 50, seed=3), limited(data, 50, seed=3)
    assert len(a.y_train) == 50 and np.array_equal(a.y_train, b.y_train)
    assert limited(data, None, seed=3) is data


def test_model_survives_the_flower_array_record_round_trip():
    model = SmallCNN(3, 5)
    record = flwr_app.ArrayRecord(model.state_dict())
    other = SmallCNN(3, 5)
    other.load_state_dict(record.to_torch_state_dict())
    for (name, p), (_, q) in zip(
        model.state_dict().items(), other.state_dict().items(), strict=True
    ):
        assert torch.equal(p, q), name


@pytest.mark.skipif(
    not (config.DATA_DIR / "raw" / "bloodmnist.npz").exists(), reason="needs the local dataset"
)
def test_load_site_returns_each_sites_own_images():
    sizes = [len(load_site("bloodmnist", i, 3, 0.5, 0, 600)[1]) for i in range(3)]
    assert sum(sizes) <= 600 and min(sizes) >= 10
