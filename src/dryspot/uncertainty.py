"""Calibration and conformal prediction sets, both fitted on held-out map blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from crepes import ConformalClassifier
from sklearn.isotonic import IsotonicRegression

from .evaluate import expected_calibration_error

MIN_BIN = 60


def fit_isotonic(p, y) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(p, float), np.asarray(y, float))
    return iso


def calibration_by_halves(p, y, blocks, seed: int = 0) -> dict:
    """Fit isotonic on half the blocks, score on the other half, and swap.

    Calibration measured on the data used to fit the calibrator would flatter it.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    uniq = pd.unique(np.asarray(blocks))
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(uniq)[: len(uniq) // 2])
    in_a = np.fromiter((b in half for b in blocks), bool, count=len(blocks))
    raw, cal, brier_raw, brier_cal = [], [], [], []
    for fit_mask in (in_a, ~in_a):
        iso = fit_isotonic(p[fit_mask], y[fit_mask])
        q = iso.predict(p[~fit_mask])
        raw.append(expected_calibration_error(y[~fit_mask], p[~fit_mask]))
        cal.append(expected_calibration_error(y[~fit_mask], q))
        brier_raw.append(float(np.mean((p[~fit_mask] - y[~fit_mask]) ** 2)))
        brier_cal.append(float(np.mean((q - y[~fit_mask]) ** 2)))
    return {
        "ece_raw": float(np.mean(raw)),
        "ece_calibrated": float(np.mean(cal)),
        "brier_raw": float(np.mean(brier_raw)),
        "brier_calibrated": float(np.mean(brier_cal)),
    }


def reliability_table(p, y, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return pd.DataFrame(rows, columns=["predicted", "observed", "n"])


def _country_bins(countries, reference) -> np.ndarray:
    counts = pd.Series(np.asarray(reference)).value_counts()
    keep = set(counts[counts >= MIN_BIN].index)
    return np.array([c if c in keep else "other" for c in np.asarray(countries)], dtype=object)


def nonconformity(p) -> np.ndarray:
    """Columns are classes (0 = water, 1 = dry). Score is 1 - probability of that class."""
    p = np.asarray(p, float)
    return np.column_stack([p, 1 - p])


def conformal_report(p, y, countries, blocks, confidence: float = 0.9, seed: int = 0) -> dict:
    """Mondrian-by-country conformal sets calibrated on half the blocks, tested on the rest."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    uniq = pd.unique(np.asarray(blocks))
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(uniq)[: len(uniq) // 2])
    cal = np.fromiter((b in half for b in blocks), bool, count=len(blocks))
    test = ~cal
    bins_cal = _country_bins(np.asarray(countries)[cal], np.asarray(countries)[cal])
    keep = set(np.unique(bins_cal))
    bins_test = np.array(
        [c if c in keep else "other" for c in np.asarray(countries)[test]], dtype=object
    )
    if "other" not in keep:
        bins_test = np.array(
            [c if c in keep else next(iter(keep)) for c in bins_test], dtype=object
        )
    scores_cal = nonconformity(p[cal])[np.arange(cal.sum()), y[cal]]
    cc = ConformalClassifier().fit(scores_cal, bins=bins_cal)
    sets = cc.predict_set(nonconformity(p[test]), bins=bins_test, confidence=confidence, seed=seed)
    yt = y[test]
    covered = sets[np.arange(len(yt)), yt].astype(bool)
    size = sets.sum(axis=1)
    single = size == 1
    per_country = (
        pd.DataFrame({"country": np.asarray(countries)[test], "covered": covered})
        .groupby("country")["covered"]
        .agg(["mean", "size"])
        .query("size >= 200")
    )
    pred_single = sets[:, 1].astype(bool)  # only meaningful where the set is a single label
    single_acc = (
        float((pred_single[single] == yt[single].astype(bool)).mean())
        if single.any()
        else float("nan")
    )
    return {
        "confidence": confidence,
        "coverage": float(covered.mean()),
        "single_label_share": float(single.mean()),
        "both_labels_share": float((size == 2).mean()),
        "empty_share": float((size == 0).mean()),
        "single_label_accuracy": single_acc,
        "worst_country_coverage": float(per_country["mean"].min()),
        "worst_country": str(per_country["mean"].idxmin()),
        "median_country_coverage": float(per_country["mean"].median()),
        "countries_below_80pct": sorted(per_country.index[per_country["mean"] < 0.8].tolist()),
        "countries_checked": int(len(per_country)),
    }
