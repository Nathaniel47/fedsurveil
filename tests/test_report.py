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


def test_cross_dataset_table_covers_each_dataset_with_three_seeds(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "CROSS_TABLE", tmp_path / "c.md")
    data = report.cross_dataset(*tables)
    assert {"bloodmnist", "organamnist", "pathmnist", "ip102"} <= set(data)
    for name, settings in data.items():
        assert len(settings) == len(report.CROSS_SETTINGS)
        for label, r in settings.items():
            assert r["n"] == 3, (name, label)
            # the chart draws one dot per seed
            assert len(list(r["acc"])) == 3, (name, label)
    # the SCAFFOLD collapse is in the data, not averaged away
    assert min(data["organamnist"]["SCAFFOLD, alpha 0.1"]["acc"]) < 0.5
    text = report.write_cross_table(data)
    assert "organamnist acc" in text and "ip102 acc" in text and "± " in text


def test_dataset_chart_is_written(tables, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(report, "CHARTS", tmp_path)
    report.chart_datasets(report.cross_dataset(*tables))
    assert (tmp_path / "datasets_alpha01.png").stat().st_size > 5000
