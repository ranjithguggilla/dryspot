"""Metrics, calibration and block bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Equal-width ECE. Good enough for a headline number; the plot shows the detail."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def precision_at_share(y, p, share: float) -> float:
    """Dry share among the top `share` of points by predicted risk."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(len(y) * share)))
    top = np.argpartition(-p, k - 1)[:k]
    return float(y[top].mean())


def summarise(y, p, share: float = 0.10) -> dict:
    y = np.asarray(y)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "precision_at_share": precision_at_share(y, p, share),
    }


def lift_curve(y, p, shares=(0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    base = float(np.mean(y))
    rows = [(s, precision_at_share(y, p, s)) for s in shares]
    out = pd.DataFrame(rows, columns=["share_visited", "precision"])
    out["lift"] = out["precision"] / base
    return out


def block_bootstrap(
    y, p, groups, stat, n_boot: int = 200, seed: int = 0
) -> tuple[float, float, float]:
    """Resample whole cells, because points inside a cell are not independent."""
    y, p = np.asarray(y), np.asarray(p)
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[bounds[g] : bounds[g + 1]] for g in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(stat(yy, p[idx]))
    point = stat(y, p)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_block_bootstrap(y, p_a, p_b, groups, stat, n_boot: int = 200, seed: int = 0):
    """Difference stat(a) - stat(b) with a cell-level bootstrap, same resamples for both."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[bounds[g] : bounds[g + 1]] for g in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(stat(yy, p_a[idx]) - stat(yy, p_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(stat(y, p_a) - stat(y, p_b)), float(lo), float(hi)
