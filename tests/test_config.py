from fedsurveil import config


def test_dermamnist_is_excluded_for_licence_reasons():
    assert "dermamnist" not in config.MEDMNIST_DATASETS


def test_algorithms_match_plan():
    assert config.ALGORITHMS == ("fedavg", "fedprox", "scaffold")


def test_dp_delta_is_small():
    assert 0 < config.DP_DELTA <= 1e-4


def test_ip102_path_comes_from_environment(monkeypatch):
    monkeypatch.delenv("IP102_NPZ_PATH", raising=False)
    assert config.ip102_npz_path() is None
    monkeypatch.setenv("IP102_NPZ_PATH", "x.npz")
    assert str(config.ip102_npz_path()) == "x.npz"


def test_env_example_has_names_only():
    lines = (config.ROOT / ".env.example").read_text().splitlines()
    for line in lines:
        if line.strip() and not line.startswith("#"):
            name, _, value = line.partition("=")
            assert name and value == "", f"{name} must have an empty value"


def test_env_is_gitignored():
    assert ".env" in (config.ROOT / ".gitignore").read_text().splitlines()
