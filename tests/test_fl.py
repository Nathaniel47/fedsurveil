import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from fedsurveil.comms.meter import state_bytes  # noqa: E402
from fedsurveil.data.load import ImageSet  # noqa: E402
from fedsurveil.fl.client import local_train, predict, to_batch  # noqa: E402
from fedsurveil.fl.fedavg import fedavg_aggregate, state_delta  # noqa: E402
from fedsurveil.fl.simulate import RunConfig, run  # noqa: E402
from fedsurveil.models.cnn import SmallCNN, SmallResNet, build_model  # noqa: E402


def synthetic(n_per_class=60, classes=4, size=16, seed=0) -> ImageSet:
    """Each class is a bright patch in a different corner, so a tiny CNN can learn it fast."""
    rng = np.random.default_rng(seed)

    def make(n):
        xs, ys = [], []
        for c in range(classes):
            x = rng.integers(0, 40, size=(n, size, size, 1), dtype=np.uint8)
            r, col = (c // 2) * (size // 2), (c % 2) * (size // 2)
            x[:, r : r + size // 2, col : col + size // 2, :] += 150
            xs.append(x)
            ys.append(np.full(n, c))
        return np.concatenate(xs), np.concatenate(ys).astype(np.int64)

    xt, yt = make(n_per_class)
    xv, yv = make(10)
    xe, ye = make(20)
    return ImageSet("synthetic", xt, yt, xv, yv, xe, ye)


def test_models_output_shapes_and_have_no_batch_norm():
    for model, shape in [
        (SmallCNN(1, 5), (2, 1, 28, 28)),
        (SmallCNN(3, 5), (2, 3, 28, 28)),
        (SmallResNet(3, 5), (2, 3, 64, 64)),
    ]:
        assert model(torch.zeros(shape)).shape == (2, 5)
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("vgg", 1, 2)


def test_to_batch_scales_to_minus_one_one_and_moves_channels_first():
    x = np.array([[[[0], [255]]]], dtype=np.uint8)
    t = to_batch(x, "cpu")
    assert t.shape == (1, 1, 1, 2)
    assert t.min().item() == -1.0 and t.max().item() == 1.0


def test_fedavg_aggregate_is_the_size_weighted_mean():
    g = {"w": torch.tensor([1.0, 1.0])}
    deltas = [{"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([0.0, 4.0])}]
    out = fedavg_aggregate(g, deltas, sizes=[1, 3])
    assert torch.allclose(out["w"], torch.tensor([1.25, 4.0]))


def test_fedavg_with_one_site_equals_that_sites_model():
    old = {"w": torch.tensor([1.0, 2.0])}
    new = {"w": torch.tensor([3.0, -1.0])}
    out = fedavg_aggregate(old, [state_delta(new, old)], [10])
    assert torch.allclose(out["w"], new["w"])


def test_fedavg_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        fedavg_aggregate({"w": torch.zeros(1)}, [], [])


def test_local_train_learns_and_is_deterministic():
    data = synthetic()
    kwargs = dict(epochs=8, lr=0.05, momentum=0.9, batch_size=32, seed=1)
    torch.manual_seed(0)
    a = SmallCNN(1, 4)
    b = SmallCNN(1, 4)
    b.load_state_dict(a.state_dict())
    loss_a = local_train(a, data.x_train, data.y_train, **kwargs)
    loss_b = local_train(b, data.x_train, data.y_train, **kwargs)
    assert loss_a == loss_b
    assert all(
        torch.equal(p, q)
        for p, q in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )
    assert (predict(a, data.x_test) == data.y_test).mean() > 0.6


def read_log(path):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return lines[0], lines[1:]


def test_federated_run_logs_meta_rounds_bytes_and_learns(tmp_path):
    cfg = RunConfig(
        dataset="synthetic", num_sites=4, alpha=5.0, rounds=5, local_epochs=2,
        lr=0.05, momentum=0.9, batch_size=32, site_eval_size=40, min_site_size=5,
    )  # fmt: skip
    path = run(cfg, data=synthetic(), device="cpu", out_dir=tmp_path)
    meta, rounds = read_log(path)
    assert meta["type"] == "meta" and meta["config"]["num_sites"] == 4
    assert sum(meta["site_sizes"]) == 240
    assert len(rounds) == 5
    assert len(rounds[-1]["per_site_acc"]) == 4
    assert rounds[0]["bytes_up"] == 4 * meta["model_bytes"]
    assert rounds[-1]["global_acc"] > 0.6


def test_partial_participation_sends_fewer_bytes(tmp_path):
    cfg = RunConfig(
        dataset="synthetic", num_sites=4, alpha=5.0, rounds=2, sites_per_round=2,
        batch_size=32, site_eval_size=20, min_site_size=5,
    )  # fmt: skip
    meta, rounds = read_log(run(cfg, data=synthetic(), device="cpu", out_dir=tmp_path))
    assert rounds[0]["bytes_up"] == 2 * meta["model_bytes"]


def test_same_seed_gives_identical_logs(tmp_path):
    cfg = RunConfig(
        dataset="synthetic", num_sites=3, alpha=1.0, rounds=2, batch_size=32,
        site_eval_size=20, min_site_size=5,
    )  # fmt: skip
    a = run(cfg, data=synthetic(), device="cpu", out_dir=tmp_path / "a")
    b = run(cfg, data=synthetic(), device="cpu", out_dir=tmp_path / "b")
    assert read_log(a)[1] == read_log(b)[1]


def test_central_baseline_has_no_traffic_and_learns(tmp_path):
    cfg = RunConfig(
        dataset="synthetic", algorithm="central", num_sites=3, alpha=1.0, rounds=4,
        local_epochs=2, lr=0.05, momentum=0.9, batch_size=32, site_eval_size=20, min_site_size=5,
    )  # fmt: skip
    _, rounds = read_log(run(cfg, data=synthetic(), device="cpu", out_dir=tmp_path))
    assert rounds[-1]["bytes_up"] == 0
    assert rounds[-1]["global_acc"] > 0.6


def test_state_bytes_counts_four_bytes_per_float():
    assert state_bytes({"a": torch.zeros(10), "b": torch.zeros(2, 3)}) == 64
