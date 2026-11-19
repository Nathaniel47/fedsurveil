import pytest

from fedsurveil.eval.comms_report import (
    analytic_table,
    build_report,
    cost_to_target,
    per_site_round_bytes,
    seconds_per_round,
    target_table,
)
from fedsurveil.eval.runs import load_log
from tests.test_runs import write_log


def fixture_run(tmp_path, **kw):
    p = tmp_path / "toy_fedavg_sites2_alpha0.5_seed0.jsonl"
    write_log(p, **kw)
    return load_log(p)


def test_per_site_bytes_divide_the_round_totals(tmp_path):
    run = fixture_run(tmp_path)
    assert per_site_round_bytes(run) == (1000, 1000)


def test_seconds_per_round_is_transfer_time_plus_latency(tmp_path):
    run = fixture_run(tmp_path)
    # (1000 + 1000) bytes * 8 bits / 8000 bit/s = 2 s, plus 2 transfers * 0.5 s latency
    assert seconds_per_round(run, 8000.0, latency_s=0.5) == pytest.approx(3.0)


def test_cost_to_target_counts_rounds_megabytes_and_hours(tmp_path):
    run = fixture_run(tmp_path)  # accuracy is 0.3, 0.6, 0.9
    cost = cost_to_target(run, 0.5)
    assert cost["rounds"] == 2
    assert cost["total MB"] == pytest.approx(0.008)
    assert cost["hours, 2G, 0.1 Mbit/s"] > cost["hours, 4G, 10 Mbit/s"] > 0
    assert cost_to_target(run, 0.95) is None


def test_target_table_skips_runs_that_send_nothing(tmp_path):
    central = tmp_path / "toy_central_sites2_alpha0.5_seed0.jsonl"
    write_log(central, algorithm="central")
    text = central.read_text().replace('"bytes_up": 2000', '"bytes_up": 0')
    central.write_text(text)
    table = target_table([load_log(central), fixture_run(tmp_path)], 0.5)
    assert len(table) == 1


def test_analytic_table_matches_the_documented_sizes():
    table = analytic_table(model_bytes=4000)  # 1000 weights
    up = dict(zip(table["scheme"], table["upload bytes per round"], strict=True))
    assert up["dense float32 (FedAvg, FedProx)"] == 4000
    assert up["int8 quantization"] == 1004
    assert up["top-k, keep 10%"] == 800
    assert up["top-k, keep 1%"] == 80
    assert up["SCAFFOLD (dense, update + control variate)"] == 8000


def test_report_has_both_parts_and_states_the_assumptions(tmp_path):
    text = build_report([fixture_run(tmp_path)], targets=(0.5,))
    assert "illustrative" in text and "Cost to reach 50% accuracy" in text
    assert "Bytes one site uploads per round" in text and "|" in text
    assert "No compressed run has been logged yet" in text
