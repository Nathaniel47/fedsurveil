import json

import pytest

torch = pytest.importorskip("torch")

from fedsurveil.fl.client import local_train, num_steps  # noqa: E402
from fedsurveil.fl.scaffold import (  # noqa: E402
    effective_lr,
    update_server_control,
    update_site_control,
    zero_controls,
)
from fedsurveil.fl.simulate import RunConfig, run, run_name  # noqa: E402
from fedsurveil.models.cnn import SmallCNN  # noqa: E402
from tests.test_fl import read_log, synthetic  # noqa: E402


def trained(seed=0, **kwargs):
    data = synthetic(n_per_class=20)
    torch.manual_seed(0)
    model = SmallCNN(1, 4)
    start = [p.detach().clone() for p in model.parameters()]
    local_train(
        model, data.x_train, data.y_train, epochs=2, lr=0.05, batch_size=16, seed=seed, **kwargs
    )
    return start, [p.detach() for p in model.parameters()]


def distance(a, b):
    return float(sum(((x - y) ** 2).sum() for x, y in zip(a, b, strict=True)) ** 0.5)


def test_fedprox_pulls_the_model_back_toward_the_start():
    start, free = trained()
    _, pulled = trained(prox_mu=5.0)
    assert distance(start, pulled) < distance(start, free)


def test_prox_mu_zero_changes_nothing():
    _, a = trained()
    _, b = trained(prox_mu=0.0)
    assert distance(a, b) == 0.0


def test_scaffold_with_zero_controls_equals_plain_training():
    model = SmallCNN(1, 4)
    zeros = zero_controls(model)
    _, plain = trained()
    _, corrected = trained(control=(zeros, zeros))
    assert distance(plain, corrected) == 0.0


def test_scaffold_correction_changes_training_when_controls_differ():
    model = SmallCNN(1, 4)
    zeros = zero_controls(model)
    ones = [torch.ones_like(c) * 0.5 for c in zeros]
    _, plain = trained()
    _, corrected = trained(control=(ones, zeros))
    assert distance(plain, corrected) > 0.0


def test_site_control_update_matches_the_formula():
    new, delta = update_site_control(
        c_site=[torch.tensor([0.0])],
        c_server=[torch.tensor([0.0])],
        global_params=[torch.tensor([1.0])],
        local_params=[torch.tensor([0.5])],
        steps=5,
        lr_eff=0.1,
    )
    assert torch.allclose(new[0], torch.tensor([1.0]))
    assert torch.allclose(delta[0], torch.tensor([1.0]))


def test_server_control_averages_over_all_sites_not_only_the_sampled_ones():
    out = update_server_control([torch.tensor([1.0])], [[torch.tensor([2.0])]], num_sites=4)
    assert torch.allclose(out[0], torch.tensor([1.5]))


def test_effective_lr_with_momentum():
    assert effective_lr(0.02, 0.0) == 0.02
    assert abs(effective_lr(0.02, 0.9) - 0.2) < 1e-9


def test_num_steps_counts_a_short_last_batch():
    assert num_steps(100, 32, 2) == 8


def test_run_name_includes_mu_for_fedprox():
    assert "fedprox-mu0.1" in run_name(RunConfig(algorithm="fedprox", prox_mu=0.1))


def test_fedprox_requires_a_positive_mu(tmp_path):
    with pytest.raises(ValueError):
        run(RunConfig(algorithm="fedprox"), data=synthetic(), out_dir=tmp_path)


def test_unknown_algorithm_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run(RunConfig(algorithm="fedsgd"), data=synthetic(), out_dir=tmp_path)


def cfg(algorithm, **kw):
    base = dict(
        dataset="synthetic", algorithm=algorithm, num_sites=4, alpha=5.0, rounds=4,
        local_epochs=2, lr=0.05, momentum=0.9, batch_size=32, site_eval_size=40,
        min_site_size=5,
    )  # fmt: skip
    base.update(kw)
    return RunConfig(**base)


def test_one_site_scaffold_equals_fedavg(tmp_path):
    kw = dict(num_sites=1, momentum=0.0, rounds=3, min_site_size=5)
    a = run(cfg("fedavg", **kw), data=synthetic(), device="cpu", out_dir=tmp_path / "a")
    b = run(cfg("scaffold", **kw), data=synthetic(), device="cpu", out_dir=tmp_path / "b")
    accs = [[r["global_acc"] for r in read_log(p)[1]] for p in (a, b)]
    assert accs[0] == accs[1]


def test_scaffold_sends_twice_the_bytes_of_fedavg(tmp_path):
    a = run(cfg("fedavg", rounds=1), data=synthetic(), device="cpu", out_dir=tmp_path / "a")
    b = run(cfg("scaffold", rounds=1), data=synthetic(), device="cpu", out_dir=tmp_path / "b")
    assert read_log(b)[1][0]["bytes_up"] == 2 * read_log(a)[1][0]["bytes_up"]


@pytest.mark.parametrize("algorithm,extra", [("fedprox", {"prox_mu": 0.1}), ("scaffold", {})])
def test_new_algorithms_learn_on_toy_data(tmp_path, algorithm, extra):
    path = run(cfg(algorithm, **extra), data=synthetic(), device="cpu", out_dir=tmp_path)
    meta, rounds = read_log(path)
    assert meta["config"]["algorithm"] == algorithm
    assert rounds[-1]["global_acc"] > 0.5
    json.dumps(rounds[-1])
