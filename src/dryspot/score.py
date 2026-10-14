"""Scoring functions used by the command line, the web app and the tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from . import config, explain
from .features import RAW_INPUT
from .models import predict_dry

RISK_BANDS = [
    (0.0, 0.15, "low"),
    (0.15, 0.35, "moderate"),
    (0.35, 0.60, "high"),
    (0.60, 1.01, "very high"),
]


def load_bundle(path: Path | str | None = None) -> dict:
    path = Path(path) if path else config.OUTPUT_DIR / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `dryspot train` first.")
    return joblib.load(path)


def band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "very high"


def normalise_input(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure every raw column the pipeline expects exists, filling gaps with missing."""
    out = df.copy()
    for col in RAW_INPUT:
        if col not in out.columns:
            out[col] = pd.NA
    for col in ("lat_deg", "lon_deg", "install_year"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["report_date"].isna().all():
        out["report_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    out["report_date"] = out["report_date"].fillna(pd.Timestamp.today().strftime("%Y-%m-%d"))
    return out


def score_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add calibrated risk, a band and a visit rank (1 is the most urgent)."""
    x = normalise_input(df)
    raw = predict_dry(bundle["model"], x)
    cal = bundle.get("calibrator")
    risk = cal.predict(raw) if cal is not None else raw
    out = df.copy()
    out["risk"] = risk.round(4)
    out["band"] = [band(p) for p in risk]
    out["visit_rank"] = (
        pd.Series(risk, index=out.index).rank(ascending=False, method="first").astype(int)
    )
    return out.sort_values("visit_rank")


def score_one(bundle: dict, record: dict, n_reasons: int = 3) -> dict:
    """Risk for one water point plus plain-language reasons."""
    row = normalise_input(pd.DataFrame([record]))
    scored = score_frame(bundle, row)
    reasons = explain.reasons(bundle["model"], row, bundle["background"], top=n_reasons)
    return {
        "risk": float(scored["risk"].iloc[0]),
        "band": scored["band"].iloc[0],
        "reasons": [explain.sentence(r, row) for r in reasons],
    }


def triage_split(scored: pd.DataFrame, share: float = config.DEFAULT_VISIT_SHARE) -> pd.DataFrame:
    """Mark the top `share` of a list as the visit-first group."""
    k = max(1, int(round(len(scored) * share)))
    out = scored.copy()
    out["visit_first"] = out["visit_rank"] <= k
    return out
