import pytest

torch = pytest.importorskip("torch")

from fedsurveil.comms.compress import (  # noqa: E402
    UpdateCompressor,
    quantize_int8,
    sparsify_topk,
)
from fedsurveil.fl.simulate import RunConfig, run, run_name  # noqa: E402
from tests.test_fl import read_log, synthetic  # noqa: E402


def test_int8_error_is_at_most_half_a_step_and_size_is_one_byte_per_weight():
    g = torch.Generator().manual_seed(0)
    v = torch.randn(1000, generator=g)
    decoded, nbytes = quantize_int8(v)
    step = float(v.abs().max()) / 127
    assert float((decoded - v).abs().max()) <= step / 2 + 1e-6
    assert nbytes == 1000 + 4


def test_int8_handles_an_all_zero_update():
    decoded, _ = quantize_int8(torch.zeros(10))
    assert torch.equal(decoded, torch.zeros(10))


def test_topk_keeps_the_largest_entries_and_zeroes_the_rest():
    v = torch.tensor([0.1, -5.0, 0.2, 3.0, -0.05])
    decoded, nbytes = sparsify_topk(v, 0.4)
    assert decoded.tolist() == [0.0, -5.0, 0.0, 3.0, 0.0]
    assert nbytes == 2 * 8


def test_topk_always_keeps_at_least_one_entry():
    decoded, nbytes = sparsify_topk(torch.tensor([1.0, 2.0, 3.0]), 0.0001)
    assert int((decoded != 0).sum()) == 1 and nbytes == 8


def test_error_feedback_never_loses_update_mass():
    comp = UpdateCompressor("topk", num_sites=1, topk_fraction=0.2)
    update = torch.arange(1.0, 11.0)
    sent = torch.zeros(10)
    for _ in range(6):
        decoded, _ = comp(0, update)
        sent += decoded
    total_put_in = update * 6
    assert torch.allclose(sent + comp.residual[0], total_put_in)
    assert int((sent > 0).sum()) > 2  # entries dropped early get sent in later rounds


def test_sites_keep_separate_residuals():
    comp = UpdateCompressor("topk", num_sites=2, topk_fraction=0.5)
    comp(0, torch.tensor([1.0, 0.1]))
    comp(1, torch.tensor([0.1, 1.0]))
    assert not torch.equal(comp.residual[0], comp.residual[1])


def test_none_is_the_identity_and_costs_dense_bytes():
    v = torch.randn(7)
    out, nbytes = UpdateCompressor("none", 1)(0, v)
    assert torch.equal(out, v) and nbytes == 28


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        UpdateCompressor("zip", 1)
    with pytest.raises(ValueError):
        UpdateCompressor("topk", 1, topk_fraction=0.0)


def cfg(**kw):
    base = dict(
        dataset="synthetic", num_sites=4, alpha=5.0, rounds=3, local_epochs=2, lr=0.05,
        momentum=0.9, batch_size=32, site_eval_size=40, min_site_size=5,
    )  # fmt: skip
    base.update(kw)
    return RunConfig(**base)


def go(tmp_path, **kw):
    return read_log(run(cfg(**kw), data=synthetic(), device="cpu", out_dir=tmp_path))


def test_int8_run_uploads_about_a_quarter_of_the_dense_bytes_and_still_learns(tmp_path):
    meta, dense = go(tmp_path / "a")
    _, small = go(tmp_path / "b", compress="int8")
    n_values = meta["model_bytes"] // 4
    assert small[0]["bytes_up"] == 4 * (n_values + 4)
    assert small[0]["bytes_down"] == dense[0]["bytes_down"]
    assert dense[0]["bytes_up"] / small[0]["bytes_up"] > 3.9
    assert small[-1]["global_acc"] > 0.5


def test_topk_run_uploads_k_values_and_positions_per_site(tmp_path):
    meta, rounds = go(tmp_path, compress="topk", topk_fraction=0.1)
    n_values = meta["model_bytes"] // 4
    assert rounds[0]["bytes_up"] == 4 * round(n_values * 0.1) * 8


def test_run_names_separate_compression_settings():
    names = {
        run_name(cfg()),
        run_name(cfg(compress="int8")),
        run_name(cfg(compress="topk", topk_fraction=0.1)),
        run_name(cfg(compress="topk", topk_fraction=0.01)),
    }
    assert len(names) == 4


def test_compression_is_rejected_where_it_does_not_apply(tmp_path):
    for kw in [
        dict(compress="int8", secagg=True),
        dict(compress="int8", algorithm="scaffold"),
        dict(compress="int8", algorithm="central"),
    ]:
        with pytest.raises(ValueError):
            run(cfg(**kw), data=synthetic(), device="cpu", out_dir=tmp_path)
