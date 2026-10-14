import numpy as np

from dryspot import evaluate as ev


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert ev.expected_calibration_error(y, p) < 0.01


def test_ece_large_for_overconfident_scores():
    y = np.array([0, 1] * 500)
    assert ev.expected_calibration_error(y, np.full(1000, 0.95)) > 0.4


def test_precision_at_share():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    assert ev.precision_at_share(y, p, 0.2) == 1.0
    assert ev.precision_at_share(y, p[::-1], 0.2) == 0.0


def test_lift_curve_is_monotone_for_a_good_ranker():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    p = y * 0.4 + rng.uniform(0, 0.6, 5000)
    lift = ev.lift_curve(y, p)
    assert lift["precision"].is_monotonic_decreasing
    assert lift["lift"].iloc[0] > 1.5


def test_block_bootstrap_interval_brackets_point():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    groups = rng.integers(0, 60, 6000)
    y = rng.integers(0, 2, 6000)
    p = y * 0.3 + rng.uniform(0, 1, 6000)
    point, lo, hi = ev.block_bootstrap(y, p, groups, roc_auc_score, n_boot=100)
    assert lo < point < hi


def test_paired_bootstrap_detects_a_better_model():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    groups = rng.integers(0, 80, 8000)
    y = rng.integers(0, 2, 8000)
    good = y * 0.8 + rng.uniform(0, 1, 8000)
    bad = rng.uniform(0, 1, 8000)
    diff, lo, hi = ev.paired_block_bootstrap(y, good, bad, groups, roc_auc_score, n_boot=100)
    assert diff > 0.2 and lo > 0
