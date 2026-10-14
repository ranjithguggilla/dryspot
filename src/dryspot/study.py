"""End-to-end study: validation regimes, model choice, calibration, economics, artefacts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config, data, economics, evaluate, explain, plots, uncertainty, validation
from .models import MODEL_NAMES, make_model, predict_dry

REGIME_ORDER = ["random", "spatial_block", "country_holdout", "forward_in_time"]
SELECTION_REGIME = "spatial_block"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_regime(df: pd.DataFrame, regime: str, models: list[str], n_splits: int, seed: int):
    """Out-of-fold probabilities for every model under one hold-out regime."""
    folds = validation.folds_for(regime, df, n_splits, seed)
    oof = {m: np.full(len(df), np.nan) for m in models}
    fold_auc = {m: [] for m in models}
    for i, (tr, te) in enumerate(folds):
        y_tr, y_te = df["no_water"].to_numpy()[tr], df["no_water"].to_numpy()[te]
        for m in models:
            model = make_model(m, seed).fit(df.iloc[tr], y_tr)
            p = predict_dry(model, df.iloc[te])
            oof[m][te] = p
            if y_te.min() != y_te.max():
                fold_auc[m].append(roc_auc_score(y_te, p))
        log(f"  {regime} fold {i + 1}/{len(folds)} done")
    return oof, fold_auc


def choose_model(y, oof: dict[str, np.ndarray], cells, n_boot: int = 200, seed: int = 0):
    """Simplest model that cannot be told apart from the best on held-out map blocks.

    "Told apart" means the paired, cell-level bootstrap interval for the AUC gap
    between the best model and the candidate excludes zero. MODEL_NAMES is ordered
    from simplest to most elaborate.
    """
    names = [m for m in MODEL_NAMES if m in oof]
    mask = ~np.isnan(oof[names[0]])
    y = np.asarray(y)[mask]
    cells = np.asarray(cells)[mask]
    auc = {m: roc_auc_score(y, oof[m][mask]) for m in names}
    best = max(auc, key=auc.get)
    gaps = {}
    for m in names:
        if m == best:
            gaps[m] = (0.0, 0.0, 0.0)
            continue
        gaps[m] = evaluate.paired_block_bootstrap(
            y, oof[best][mask], oof[m][mask], cells, roc_auc_score, n_boot, seed
        )
    chosen = best
    for m in names:
        if gaps[m][1] <= 0.0:  # lower end of (best minus candidate) reaches zero
            chosen = m
            break
    detail = {
        "best": best,
        "best_auc": float(auc[best]),
        "gap_to_best": {m: {"gap": g[0], "ci": [g[1], g[2]]} for m, g in gaps.items()},
    }
    return chosen, detail


def status_text_audit(df: pd.DataFrame, seed: int) -> dict:
    """Show why the free-text `status` field must not be a feature.

    It is written after the visit, by the person who saw the pump. A text model on
    it looks superb and would be useless before anyone has gone out.
    """
    sub = df[df["status"].fillna("").str.strip().str.len() > 2]
    sub = sub[sub["status"].astype(str).str.len() < 400]
    if len(sub) < 2000 or sub["no_water"].nunique() < 2:
        return {"rows": int(len(sub)), "auc": None}
    out_auc = []
    for tr, te in validation.folds_for("spatial_block", sub, 3, seed):
        vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=40000)
        x_tr = vec.fit_transform(sub["status"].iloc[tr].astype(str))
        clf = LogisticRegression(max_iter=300, C=2.0).fit(x_tr, sub["no_water"].iloc[tr])
        p = clf.predict_proba(vec.transform(sub["status"].iloc[te].astype(str)))[:, 1]
        out_auc.append(roc_auc_score(sub["no_water"].iloc[te], p))
    return {
        "rows": int(len(sub)),
        "share_of_labelled_rows": float(len(sub) / len(df)),
        "auc": float(np.mean(out_auc)),
    }


def run_study(
    df: pd.DataFrame,
    out_dir: Path,
    asset_dir: Path,
    n_splits: int = 4,
    seed: int = config.SEED,
    models: list[str] | None = None,
    regimes: list[str] | None = None,
    n_boot: int = 200,
    ship: bool = True,
) -> dict:
    models = models or MODEL_NAMES
    regimes = regimes or REGIME_ORDER
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    df = validation.add_cells(df)
    y = df["no_water"].to_numpy()

    rows, oof_all, fold_all = [], {}, {}
    for regime in regimes:
        log(f"regime {regime}")
        oof, fold_auc = run_regime(df, regime, models, n_splits, seed)
        oof_all[regime], fold_all[regime] = oof, fold_auc
        for m in models:
            mask = ~np.isnan(oof[m])
            s = evaluate.summarise(y[mask], oof[m][mask])
            fa = fold_auc[m]
            s.update(
                model=m,
                regime=regime,
                auc_fold_mean=float(np.mean(fa)) if fa else None,
                auc_fold_se=float(np.std(fa, ddof=1) / np.sqrt(len(fa))) if len(fa) > 1 else None,
            )
            rows.append(s)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "validation_results.csv", index=False)

    if SELECTION_REGIME in fold_all:
        chosen, why = choose_model(
            y, oof_all[SELECTION_REGIME], df["cell"].to_numpy(), n_boot=n_boot, seed=seed
        )
    else:
        chosen, why = models[-1], {"note": "selection regime not run"}
    log(f"shipping {chosen}")

    metrics: dict = {
        "data": None,
        "validation": results.to_dict(orient="records"),
        "selection": {"shipped_model": chosen, **why},
    }

    use_cal = False
    sp_mask = ~np.isnan(oof_all[SELECTION_REGIME][chosen]) if SELECTION_REGIME in oof_all else None
    if sp_mask is not None:
        p_sp = oof_all[SELECTION_REGIME][chosen][sp_mask]
        y_sp = y[sp_mask]
        blocks = df["block"].to_numpy()[sp_mask]
        cells = df["cell"].to_numpy()[sp_mask]

        auc, lo, hi = evaluate.block_bootstrap(y_sp, p_sp, cells, roc_auc_score, n_boot, seed)
        metrics["shipped_spatial_auc"] = {"auc": auc, "ci": [lo, hi]}

        cal = uncertainty.calibration_by_halves(p_sp, y_sp, blocks, seed)
        metrics["calibration"] = cal
        iso_all = uncertainty.fit_isotonic(p_sp, y_sp)
        use_cal = cal["ece_calibrated"] < cal["ece_raw"]
        metrics["calibration"]["calibrator_shipped"] = bool(use_cal)
        p_final = iso_all.predict(p_sp) if use_cal else p_sp
        metrics["conformal"] = uncertainty.conformal_report(
            p_final, y_sp, df["country_name"].to_numpy()[sp_mask], blocks, 0.9, seed
        )
        plots.calibration(uncertainty.reliability_table(p_sp, y_sp), asset_dir / "calibration.png")

        base_name = "country_rate"
        p_base = (
            oof_all[SELECTION_REGIME][base_name][sp_mask]
            if base_name in models
            else p_sp * 0 + y_sp.mean()
        )
        metrics["economics"] = economics.dispatch_table(
            y_sp, p_sp, p_base, cells, n_boot=n_boot, seed=seed
        )
        sweep = economics.share_sweep(y_sp, p_sp, p_base)
        sweep.to_csv(out_dir / "precision_sweep.csv", index=False)
        plots.precision_curve(sweep, asset_dir / "precision_curve.png")
        metrics["lift"] = evaluate.lift_curve(y_sp, p_sp).to_dict(orient="records")

        per_year = (
            pd.DataFrame({"year": df["report_year"].to_numpy()[sp_mask], "y": y_sp, "p": p_sp})
            .groupby("year")
            .agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
        )
        metrics["by_report_year"] = (
            per_year[per_year.n >= 2000].reset_index().to_dict(orient="records")
        )

        oof_frame = pd.DataFrame(
            {
                "row_id": df["row_id"].to_numpy()[sp_mask],
                "y": y_sp,
                "p_raw": p_sp,
                "p_cal": iso_all.predict(p_sp),
                "country": df["country_name"].to_numpy()[sp_mask],
            }
        )
        oof_frame.to_parquet(out_dir / "oof_predictions.parquet", index=False)

    if "country_holdout" in oof_all:
        mask = ~np.isnan(oof_all["country_holdout"][chosen])
        c = pd.DataFrame(
            {
                "country": df["country_name"].to_numpy()[mask],
                "y": y[mask],
                "p": oof_all["country_holdout"][chosen][mask],
            }
        )
        tab = []
        for name, g in c.groupby("country"):
            if len(g) >= 500 and g.y.nunique() == 2:
                tab.append((name, len(g), g.y.mean(), roc_auc_score(g.y, g.p)))
        tab = pd.DataFrame(tab, columns=["country", "n", "dry_rate", "auc"])
        tab.to_csv(out_dir / "country_holdout_auc.csv", index=False)
        metrics["country_holdout"] = {
            "countries": int(len(tab)),
            "median_auc": float(tab.auc.median()),
            "worst": tab.sort_values("auc").head(3).to_dict(orient="records"),
            "best": tab.sort_values("auc").tail(3).to_dict(orient="records"),
        }
        plots.country_auc(tab, asset_dir / "country_holdout.png")

    metrics["status_text_audit"] = status_text_audit(df, seed)

    plots.validation_gap(results, asset_dir / "validation_gap.png")
    plots.dry_map(df, asset_dir / "dry_map.png")

    if ship:
        log("fitting the shipped model on all rows")
        final = make_model(chosen, seed).fit(df, y)
        # permutation importance on one spatial fold of a model that has not seen it
        tr, te = validation.folds_for(SELECTION_REGIME, df, n_splits, seed)[0]
        probe = make_model(chosen, seed).fit(df.iloc[tr], y[tr])
        te = np.random.default_rng(seed).permutation(te)[:40000]
        imp = explain.permutation_importance(probe, df.iloc[te], y[te], repeats=3, seed=seed)
        imp.to_csv(out_dir / "permutation_importance.csv", index=False)
        metrics["importance"] = imp.to_dict(orient="records")
        plots.importance(imp, asset_dir / "importance.png")

        bundle = {
            "model": final,
            "calibrator": iso_all if (sp_mask is not None and use_cal) else None,
            "background": df.sample(min(3000, len(df)), random_state=seed)[
                list(_raw_cols())
            ].reset_index(drop=True),
            "meta": {
                "shipped_model": chosen,
                "countries": sorted(df["country_name"].unique()),
                "water_sources": sorted(df["water_source"].dropna().unique()),
                "water_techs": sorted(df["water_tech"].dropna().unique()),
                "facility_types": sorted(df["facility_type"].dropna().unique()),
                "median_coords": df.groupby("country_name")[["lat_deg", "lon_deg"]]
                .median()
                .to_dict("index"),
                "rows_trained": int(len(df)),
            },
        }
        joblib.dump(bundle, out_dir / "model.joblib", compress=3)
    return metrics


def _raw_cols():
    from .features import RAW_INPUT

    return RAW_INPUT


def _sanitise(obj):
    """Replace NaN and infinity with None so the file is strict JSON."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def write_metrics(metrics: dict, path: Path) -> None:
    def default(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(_sanitise(metrics), indent=2, default=default, allow_nan=False))


def main(
    max_rows: int | None = None,
    n_splits: int = 4,
    n_boot: int = 200,
    out_dir: Path = config.OUTPUT_DIR,
    asset_dir: Path = config.ASSET_DIR,
) -> dict:
    raw = data.load_raw()
    df, tally = data.clean(raw)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=config.SEED).reset_index(drop=True)
        tally["subsampled_to"] = int(len(df))
    metrics = run_study(df, out_dir, asset_dir, n_splits=n_splits, n_boot=n_boot)
    metrics["data"] = tally
    write_metrics(metrics, out_dir / "metrics.json")
    return metrics
