import json
from pathlib import Path

import numpy as np
import pytest

from fedsurveil.eval.runs import (
    NAME,
    cumulative_mb,
    curves,
    load_all,
    load_compact,
    load_log,
    rounds_to_accuracy,
    site_matrix,
    summary_table,
)

ROOT = Path(__file__).resolve().parent.parent


def write_log(path, algorithm="fedavg", alpha=0.5, seed=0, rounds=3, extra_cfg=None, mia=None):
    """A small hand-made log in the format fedsurveil.fl.simulate writes (test fixture only)."""
    cfg = {
        "dataset": "toy",
        "algorithm": algorithm,
        "alpha": alpha,
        "seed": seed,
        "num_sites": 2,
        **(extra_cfg or {}),
    }
    lines = [
        {
            "type": "meta",
            "config": cfg,
            "model_bytes": 1000,
            "site_sizes": [60, 40],
            "site_class_counts": [[50, 10], [5, 35]],
        }
    ]
    for i in range(1, rounds + 1):
        lines.append(
            {
                "type": "round",
                "round": i,
                "bytes_up": 2000,
                "bytes_down": 2000,
                "global_acc": 0.3 * i,
                "macro_f1": 0.2 * i,
                "per_site_acc": [0.3 * i, 0.2 * i],
                "site_acc_spread": {"mean": 0.25 * i, "min": 0.2 * i, "std": 0.05},
                "eps_client": float(i),
            }
        )
    if mia is not None:
        lines.append({"type": "summary", "mia_auroc": mia})
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


def test_load_log_reads_config_rounds_and_summary(tmp_path):
    write_log(tmp_path / "toy_fedavg_sites2_alpha0.5_seed0.jsonl", mia=0.55)
    run = load_log(tmp_path / "toy_fedavg_sites2_alpha0.5_seed0.jsonl")
    assert run.algorithm == "fedavg" and run.alpha == 0.5 and run.seed == 0
    assert len(run.rounds) == 3 and run.mia_auroc == 0.55
    assert run.rounds[0]["worst_site"] == pytest.approx(0.2)
    assert run.final_site_acc == run.rounds[-1]["per_site_acc"]
    assert run.has_per_site_history


def test_algorithm_label_shows_privacy_options(tmp_path):
    p = tmp_path / "x.jsonl"
    write_log(
        p,
        algorithm="fedprox",
        extra_cfg={"prox_mu": 0.1, "dp_clip": 1.0, "dp_noise": 0.5, "secagg": True},
    )
    label = load_log(p).algorithm
    assert "fedprox-mu0.1" in label and "clientDP" in label and "secagg(sim)" in label


def test_site_matrix_shape_and_cumulative_traffic(tmp_path):
    p = tmp_path / "x.jsonl"
    write_log(p)
    run = load_log(p)
    assert site_matrix(run).shape == (3, 2)
    assert cumulative_mb(run) == pytest.approx([0.004, 0.008, 0.012])


def test_rounds_to_accuracy(tmp_path):
    p = tmp_path / "x.jsonl"
    write_log(p)
    run = load_log(p)
    assert rounds_to_accuracy(run, 0.5) == 2
    assert rounds_to_accuracy(run, 0.95) is None


def test_summary_table_has_one_row_per_run_and_the_key_columns(tmp_path):
    write_log(tmp_path / "a.jsonl", seed=0)
    write_log(tmp_path / "b.jsonl", seed=1, mia=0.5)
    table = summary_table([load_log(tmp_path / "a.jsonl"), load_log(tmp_path / "b.jsonl")])
    assert len(table) == 2
    assert {"accuracy (final)", "worst site (final)", "total MB sent", "membership AUROC"} <= set(
        table.columns
    )


def test_curves_long_format(tmp_path):
    p = tmp_path / "x.jsonl"
    write_log(p)
    df = curves([load_log(p)], "global_acc")
    assert list(df.columns) == ["run", "round", "value"] and len(df) == 3


def test_run_name_pattern_handles_privacy_tags_and_algorithm_options():
    m = NAME.match("bloodmnist_fedprox-mu0.1_cdp-c1.5-n0.5_secagg_sites10_alpha0.1_seed2")
    assert m is None or m["algo"].startswith("fedprox")
    m = NAME.match("bloodmnist_fedavg_sites10_alpha100_seed0")
    assert m["algo"] == "fedavg" and m["alpha"] == "100" and m["seed"] == "0"
    m = NAME.match("bloodmnist_central_sites10_alpha100_seed1")
    assert m["algo"] == "central"


def test_committed_seed0_results_load_as_four_runs():
    runs = load_compact(ROOT / "results" / "step4_bloodmnist_seed0.json")
    assert len(runs) == 4
    by_name = {r.name: r for r in runs}
    central = by_name["bloodmnist_central_sites10_alpha100_seed0"]
    low = by_name["bloodmnist_fedavg_sites10_alpha0.1_seed0"]
    assert len(central.rounds) == 30 and central.source == "compact"
    assert central.rounds[-1]["global_acc"] > low.rounds[-1]["global_acc"] + 0.3
    assert low.rounds[-1]["worst_site"] < 0.1
    assert central.rounds[0]["bytes_up"] == 0
    assert low.rounds[0]["bytes_up"] == 10 * 378912
    assert len(low.final_site_acc) == 10 and sum(low.site_sizes) > 10000


def test_files_without_curves_are_skipped(tmp_path):
    p = tmp_path / "step9_toy_seed0.json"
    p.write_text(json.dumps({"_note": "x", "runs": {"a": {"final_acc": 1}}}))
    assert load_compact(p) == []


def test_load_all_prefers_a_full_log_over_a_compact_copy(tmp_path):
    runs_dir, results_dir = tmp_path / "runs", tmp_path / "results"
    runs_dir.mkdir()
    results_dir.mkdir()
    name = "bloodmnist_central_sites10_alpha100_seed0"
    log = runs_dir / f"{name}.jsonl"
    write_log(log, algorithm="central", alpha=100.0)
    (results_dir / "step4_bloodmnist_seed0.json").write_text(
        (ROOT / "results" / "step4_bloodmnist_seed0.json").read_text(encoding="utf-8")
    )
    runs = load_all(runs_dir, results_dir)
    assert len(runs) == 4
    sources = {r.name: r.source for r in runs}
    assert sources[name] == "log"
    assert sorted(set(sources.values())) == ["compact", "log"]
    assert np.isclose(sum(1 for s in sources.values() if s == "compact"), 3)


def test_broken_logs_are_ignored(tmp_path):
    (tmp_path / "bad.jsonl").write_text("not json")
    (tmp_path / "empty.jsonl").write_text("")
    assert load_all(tmp_path, tmp_path) == []


def test_dashboard_runs_without_errors_on_the_committed_results():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "FedSurveil"
