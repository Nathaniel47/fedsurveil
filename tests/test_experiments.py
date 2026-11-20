import pytest

from fedsurveil.fl.experiments import dataset_runs, group_name


def flag(args, name):
    return args[args.index(name) + 1]


def test_plan_has_24_runs_with_fixed_seeds_and_the_step4_settings():
    runs = dataset_runs("organamnist")
    assert len(runs) == 24  # (1 central + 3 fedavg + 2 fedprox + 2 scaffold) x 3 seeds
    assert {flag(r, "--seed") for r in runs} == {"0", "1", "2"}
    for r in runs:
        assert flag(r, "--dataset") == "organamnist"
        assert (flag(r, "--rounds"), flag(r, "--lr"), flag(r, "--local-epochs")) == (
            "30",
            "0.02",
            "2",
        )
    algos = [flag(r, "--algorithm") for r in runs if flag(r, "--seed") == "0"]
    assert sorted(algos) == ["central"] + ["fedavg"] * 3 + ["fedprox"] * 2 + ["scaffold"] * 2


def test_fedprox_always_has_a_strength_and_no_run_uses_privacy_or_compression():
    for r in dataset_runs("pathmnist"):
        if flag(r, "--algorithm") == "fedprox":
            assert flag(r, "--prox-mu") == "0.1"
        assert not {"--dp-clip", "--dp-noise", "--secagg", "--compress"} & set(r)


def test_group_names_match_the_step4_style():
    assert group_name({"algorithm": "central", "alpha": 0.5}) == "central"
    assert group_name({"algorithm": "fedavg", "alpha": 100.0}) == "fedavg_a100"
    assert (
        group_name({"algorithm": "fedprox", "alpha": 0.1, "prox_mu": 0.1}) == "fedprox-mu0.1_a0.1"
    )


def test_summary_of_a_real_tiny_run(tmp_path):
    pytest.importorskip("torch")
    from test_fl import synthetic

    from fedsurveil.fl.experiments import summarize_log
    from fedsurveil.fl.simulate import RunConfig, run

    cfg = RunConfig(
        dataset="synthetic", num_sites=3, alpha=1.0, rounds=6, batch_size=32,
        site_eval_size=20, min_site_size=5,
    )  # fmt: skip
    path = run(cfg, data=synthetic(), device="cpu", out_dir=tmp_path)
    s = summarize_log(path)
    assert s["group"] == "fedavg_a1" and s["rounds"] == 6 and s["seed"] == 0
    assert 0 <= s["final_worst_site_acc"] <= 1 and 0 <= s["final_acc"] <= 1
    assert s["bytes_up_round"] > 0
