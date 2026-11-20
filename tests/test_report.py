import pytest

from fedsurveil.eval import report


@pytest.fixture(scope="module")
def tables():
    runs = report.load_runs()
    return runs, report.summarize(runs)


def test_every_setting_has_three_seeds_except_the_single_dp_sgd_run(tables):
    _, summary = tables
    one_seed = summary[summary["seeds"] == 1]["setting"].tolist()
    assert len(one_seed) == 1 and "DP-SGD" in one_seed[0]
    assert (summary["seeds"].isin([1, 3])).all()


def test_baselines_match_the_step4_note(tables):
    _, s = tables
    fedavg = s[s["key"] == "s4_fedavg_a0.1"].iloc[0]
    assert fedavg["acc"] == pytest.approx(0.6469, abs=1e-3)
    assert fedavg["worst"] == pytest.approx(0.0987, abs=1e-3)


def test_privacy_rows_say_which_guarantee_they_give(tables):
    _, s = tables
    private = s[s["key"].str.contains("cdp|sdp")]
    assert len(private) == 4
    for g in private["guarantee"]:
        assert g.startswith(("client-level", "sample-level"))
    secagg = s[s["key"].str.contains("secagg")]["guarantee"].iloc[0]
    assert "simulation" in secagg and "not cryptographic" in secagg


def test_table_file_is_written_and_marks_single_seed(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "TABLE", tmp_path / "t.md")
    text = report.write_table(tables[1])
    assert (tmp_path / "t.md").read_text(encoding="utf-8") == text
    assert "(1 seed)" in text and "± " in text


def test_charts_are_written(tables, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(report, "CHARTS", tmp_path)
    runs, summary = tables
    report.chart_noniid(runs, summary)
    report.chart_privacy(summary)
    report.chart_comms(summary)
    for name in ("noniid_methods", "privacy_utility", "compression_cost"):
        assert (tmp_path / f"{name}.png").stat().st_size > 5000
