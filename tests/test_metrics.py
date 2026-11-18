import numpy as np

from fedsurveil.eval.metrics import accuracy, macro_f1, site_eval_indices, spread


def test_accuracy():
    assert accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == 0.75


def test_macro_f1_is_perfect_for_perfect_predictions():
    y = np.array([0, 1, 2, 2, 1, 0])
    assert macro_f1(y, y, 3) == 1.0


def test_macro_f1_penalises_ignoring_a_rare_class():
    y_true = np.array([0] * 90 + [1] * 10)
    always_zero = np.zeros(100, dtype=int)
    assert accuracy(y_true, always_zero) == 0.9
    assert macro_f1(y_true, always_zero, 2) < 0.5


def test_site_eval_sets_follow_each_sites_class_mix():
    y_test = np.repeat(np.arange(3), 200)
    counts = np.array([[100, 0, 0], [0, 50, 50]])
    idx = site_eval_indices(y_test, counts, n_per_site=400, seed=0)
    assert set(y_test[idx[0]].tolist()) == {0}
    share_1 = np.mean(y_test[idx[1]] == 1)
    assert 0.4 < share_1 < 0.6
    assert 2 not in set(y_test[idx[0]].tolist())


def test_site_eval_sets_are_deterministic():
    y_test = np.repeat(np.arange(3), 50)
    counts = np.array([[5, 3, 1], [1, 1, 8]])
    a = site_eval_indices(y_test, counts, 100, seed=7)
    b = site_eval_indices(y_test, counts, 100, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_spread_reports_worst_site():
    s = spread([0.9, 0.5, 0.7])
    assert s["min"] == 0.5
    assert abs(s["mean"] - 0.7) < 1e-9
