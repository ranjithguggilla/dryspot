import numpy as np

from dryspot import economics as ec
from dryspot import uncertainty as un


def synthetic(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.02, 0.9, n)
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    blocks = rng.integers(0, 120, n)
    countries = rng.choice(["a", "b", "c"], n)
    return p_true, y, blocks, countries


def test_isotonic_repairs_overconfident_scores():
    p_true, y, blocks, _ = synthetic()
    distorted = np.clip(0.5 + (p_true - 0.5) * 1.8, 0.01, 0.99)
    out = un.calibration_by_halves(distorted, y, blocks)
    assert out["ece_calibrated"] < out["ece_raw"]
    assert out["brier_calibrated"] <= out["brier_raw"] + 1e-6


def test_reliability_table_rows_sum_to_n():
    p_true, y, _, _ = synthetic(5000)
    tab = un.reliability_table(p_true, y)
    assert tab["n"].sum() == 5000


def test_conformal_coverage_close_to_target():
    p_true, y, blocks, countries = synthetic()
    rep = un.conformal_report(p_true, y, countries, blocks, confidence=0.9)
    assert 0.87 < rep["coverage"] < 0.94
    assert rep["single_label_accuracy"] > 0.7


def test_dispatch_table_arithmetic():
    rng = np.random.default_rng(0)
    n = 20000
    y = (rng.uniform(0, 1, n) < 0.25).astype(int)
    p_model = y * 0.5 + rng.uniform(0, 1, n)
    p_base = rng.uniform(0, 1, n)
    groups = rng.integers(0, 100, n)
    out = ec.dispatch_table(
        y, p_model, p_base, groups, share=0.1, cost_per_visit=40, visits_per_day=5, n_boot=40
    )
    prec = out["model"]["precision"]
    assert np.isclose(out["model"]["cost_per_fault_usd"], 40 / prec)
    assert np.isclose(out["model"]["tech_days_per_100_faults"], 100 / prec / 5)
    assert out["model"]["precision"] > out["random"]["precision"]
    assert out["saving_per_1000_faults_vs_random_usd"] > 0
    lo, hi = out["model"]["precision_ci"]
    assert lo <= prec <= hi


def test_share_sweep_shape():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    sweep = ec.share_sweep(y, rng.uniform(size=1000), rng.uniform(size=1000))
    assert list(sweep.columns) == ["share_visited", "random", "country_prior", "model"]
