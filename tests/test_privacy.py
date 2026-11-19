import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedsurveil.fl.fedavg import fedavg_aggregate  # noqa: E402
from fedsurveil.fl.simulate import RunConfig, run  # noqa: E402
from fedsurveil.privacy.client_dp import (  # noqa: E402
    client_level_epsilon,
    clip_vector,
    flatten,
    gaussian_rdp_epsilon,
    unflatten,
)
from fedsurveil.privacy.membership import auroc  # noqa: E402
from fedsurveil.privacy.pipeline import private_aggregate  # noqa: E402
from fedsurveil.privacy.secagg import MODULUS, decode, encode, mask_updates, secure_sum  # noqa: E402
from tests.test_fl import read_log, synthetic  # noqa: E402


def test_clip_scales_down_long_vectors_and_keeps_direction():
    v = torch.tensor([3.0, 4.0])
    clipped, norm, was = clip_vector(v, 1.0)
    assert was and norm == 5.0
    assert abs(float(clipped.norm()) - 1.0) < 1e-6
    assert torch.allclose(clipped / clipped.norm(), v / v.norm())


def test_clip_leaves_short_vectors_alone():
    v = torch.tensor([0.3, 0.4])
    clipped, _, was = clip_vector(v, 1.0)
    assert not was and torch.equal(clipped, v)


def test_flatten_unflatten_round_trip():
    state = {"a": torch.arange(6.0).reshape(2, 3), "b": torch.tensor([7.0, 8.0])}
    back = unflatten(flatten(state), state)
    assert all(torch.equal(back[k], state[k]) for k in state)


def test_epsilon_falls_with_more_noise_and_rises_with_more_rounds():
    delta = 1e-5
    assert gaussian_rdp_epsilon(2.0, 10, delta) < gaussian_rdp_epsilon(1.0, 10, delta)
    assert gaussian_rdp_epsilon(1.0, 10, delta) < gaussian_rdp_epsilon(1.0, 50, delta)
    assert math.isinf(gaussian_rdp_epsilon(0.0, 10, delta))


def test_closed_form_epsilon_matches_the_opacus_accountant_without_subsampling():
    pytest.importorskip("opacus")
    from fedsurveil.privacy.client_dp import sampled_epsilon

    ours = gaussian_rdp_epsilon(2.0, 30, 1e-5)
    theirs = sampled_epsilon([(2.0, 1.0, 30)], 1e-5)
    assert abs(ours - theirs) / theirs < 0.05


def test_partial_participation_costs_less_privacy_than_full_participation():
    pytest.importorskip("opacus")
    full = client_level_epsilon(1.5, 10, 10, 20, 1e-5)
    part = client_level_epsilon(1.5, 3, 10, 20, 1e-5)
    assert part < full


def test_fixed_point_round_trip_including_negatives():
    v = torch.tensor([0.5, -0.25, 3.0, -7.125, 0.0])
    assert torch.allclose(decode(encode(v)), v, atol=1e-4)


def test_secure_sum_equals_the_plain_sum():
    g = torch.Generator().manual_seed(0)
    vectors = [torch.randn(1000, generator=g) for _ in range(6)]
    assert torch.allclose(secure_sum(vectors, seed=3), torch.stack(vectors).sum(0), atol=1e-3)


def test_masks_cancel_exactly_in_integer_arithmetic():
    g = torch.Generator().manual_seed(1)
    vectors = [torch.randn(500, generator=g) for _ in range(4)]
    masked = mask_updates(vectors, seed=9)
    plain = sum(encode(v) for v in vectors) % MODULUS
    assert np.array_equal(np.sum(masked, axis=0) % MODULUS, plain)


def test_one_masked_update_reveals_nothing_about_the_update():
    g = torch.Generator().manual_seed(2)
    vectors = [torch.randn(5000, generator=g) for _ in range(3)]
    masked = mask_updates(vectors, seed=4)[0].astype(float)
    corr = np.corrcoef(masked, encode(vectors[0]).astype(float))[0, 1]
    assert abs(corr) < 0.1
    assert masked.std() > 0.2 * MODULUS


def test_secure_sum_with_two_sites_and_with_one_site():
    a, b = torch.tensor([1.0, -2.0]), torch.tensor([0.5, 4.0])
    assert torch.allclose(secure_sum([a, b], seed=1), a + b, atol=1e-4)
    assert torch.allclose(secure_sum([a], seed=1), a, atol=1e-4)


def make_deltas(n=4, size=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = {"w": torch.zeros(size)}
    deltas = [{"w": torch.randn(size, generator=g)} for _ in range(n)]
    return state, deltas


def test_pipeline_without_privacy_options_matches_fedavg():
    state, deltas = make_deltas()
    sizes = [10, 20, 30, 40]
    expected = fedavg_aggregate(state, deltas, sizes)
    got, _ = private_aggregate(state, deltas, sizes)
    assert torch.allclose(got["w"], expected["w"], atol=1e-6)


def test_pipeline_with_secagg_only_matches_fedavg():
    state, deltas = make_deltas()
    sizes = [10, 20, 30, 40]
    expected = fedavg_aggregate(state, deltas, sizes)
    got, _ = private_aggregate(state, deltas, sizes, secagg=True, seed=5)
    assert torch.allclose(got["w"], expected["w"], atol=1e-3)


def test_clipping_bounds_the_influence_of_each_site():
    state = {"w": torch.zeros(10)}
    deltas = [{"w": torch.full((10,), 100.0)}, {"w": torch.zeros(10)}]
    got, info = private_aggregate(state, deltas, [1, 1], clip=1.0)
    assert info["clipped_fraction"] == 0.5
    assert float(got["w"].norm()) <= 1.0 / 2 + 1e-6


def test_noise_has_the_calibrated_standard_deviation():
    size, n, clip, sigma = 40000, 5, 2.0, 1.5
    state = {"w": torch.zeros(size)}
    deltas = [{"w": torch.zeros(size)} for _ in range(n)]
    got, _ = private_aggregate(state, deltas, [1] * n, clip=clip, noise_multiplier=sigma, seed=0)
    expected_std = sigma * clip / n
    assert abs(float(got["w"].std()) - expected_std) / expected_std < 0.05


def test_noise_is_reproducible_for_a_seed_and_differs_across_seeds():
    state, deltas = make_deltas()
    kw = dict(clip=1.0, noise_multiplier=1.0)
    a, _ = private_aggregate(state, deltas, [1] * 4, seed=1, **kw)
    b, _ = private_aggregate(state, deltas, [1] * 4, seed=1, **kw)
    c, _ = private_aggregate(state, deltas, [1] * 4, seed=2, **kw)
    assert torch.equal(a["w"], b["w"]) and not torch.equal(a["w"], c["w"])


def test_auroc_known_values():
    assert auroc(np.array([3, 4, 5]), np.array([0, 1, 2])) == 1.0
    assert auroc(np.array([0, 1, 2]), np.array([3, 4, 5])) == 0.0
    assert auroc(np.array([1, 1]), np.array([1, 1])) == 0.5
    rng = np.random.default_rng(0)
    assert abs(auroc(rng.normal(size=2000), rng.normal(size=2000)) - 0.5) < 0.05


def cfg(**kw):
    base = dict(
        dataset="synthetic", num_sites=4, alpha=5.0, rounds=2, local_epochs=1, lr=0.05,
        momentum=0.9, batch_size=32, site_eval_size=40, min_site_size=5,
    )  # fmt: skip
    base.update(kw)
    return RunConfig(**base)


def go(tmp_path, **kw):
    return read_log(run(cfg(**kw), data=synthetic(), device="cpu", out_dir=tmp_path))


def test_client_dp_run_logs_a_growing_epsilon(tmp_path):
    _, rounds = go(tmp_path, dp_clip=1.0, dp_noise=1.0)
    eps = [r["eps_client"] for r in rounds]
    assert 0 < eps[0] < eps[1] < math.inf
    assert "clipped_fraction" in rounds[0]


def test_secagg_run_adds_key_exchange_bytes_and_keeps_accuracy(tmp_path):
    _, plain = go(tmp_path / "a")
    _, secure = go(tmp_path / "b", secagg=True)
    assert secure[0]["bytes_up"] == plain[0]["bytes_up"] + 4 * 3 * 32
    assert abs(secure[-1]["global_acc"] - plain[-1]["global_acc"]) <= 0.05


def test_membership_check_is_written_as_a_summary_line(tmp_path):
    path = run(cfg(mia=True), data=synthetic(), device="cpu", out_dir=tmp_path)
    last = [__import__("json").loads(line) for line in path.read_text().splitlines()][-1]
    assert last["type"] == "summary" and 0.0 <= last["mia_auroc"] <= 1.0


def test_sample_level_dp_run_logs_epsilon(tmp_path):
    pytest.importorskip("opacus")
    _, rounds = go(tmp_path, site_dp_noise=1.0, site_dp_clip=1.0, rounds=1)
    assert 0 < rounds[0]["eps_sample_max"] < math.inf


def test_invalid_privacy_combinations_are_rejected(tmp_path):
    for kw in [
        dict(algorithm="central", secagg=True),
        dict(dp_noise=1.0),
        dict(algorithm="scaffold", dp_clip=1.0),
    ]:
        with pytest.raises(ValueError):
            run(cfg(**kw), data=synthetic(), device="cpu", out_dir=tmp_path)


def test_dp_sgd_updates_the_original_model_weights():
    pytest.importorskip("opacus")
    from fedsurveil.models.cnn import SmallCNN
    from fedsurveil.privacy.dp_sgd import local_train_dp

    data = synthetic(n_per_class=20)
    torch.manual_seed(0)
    model = SmallCNN(1, 4)
    before = [p.detach().clone() for p in model.parameters()]
    history = local_train_dp(
        model, data.x_train, data.y_train, epochs=1, lr=0.05, batch_size=16, seed=0,
        noise_multiplier=1.0, max_grad_norm=1.0,
    )  # fmt: skip
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    noise, rate, steps = history[0]
    assert noise == 1.0 and 0 < rate <= 1 and steps > 0


def test_run_names_keep_privacy_settings_apart():
    from fedsurveil.fl.simulate import run_name

    names = {
        run_name(cfg()),
        run_name(cfg(dp_clip=1.0, dp_noise=0.5)),
        run_name(cfg(dp_clip=1.0, dp_noise=1.0)),
        run_name(cfg(site_dp_noise=1.0)),
        run_name(cfg(secagg=True)),
    }
    assert len(names) == 5
