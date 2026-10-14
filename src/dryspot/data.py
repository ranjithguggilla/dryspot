"""Download, load and clean the Water Point Data Exchange extract."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

LABELLED = {"y": 0, "n": 1}  # 1 means no water on the day of the visit


def download(dest: Path = config.RAW_FILE, url: str = config.RAW_URL) -> Path:
    """Fetch the csv once. A 58 MB file, so it is kept out of git."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw extract. Falls back to the committed sample for tests and demos."""
    path = Path(path) if path else config.RAW_FILE
    if not path.exists():
        path = config.SAMPLE_FILE
    return pd.read_csv(path, low_memory=False)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, vectorised."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * config.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def flag_bad_coordinates(df: pd.DataFrame) -> pd.Series:
    """True for rows whose coordinates are implausible for the country they claim.

    A point is rejected when it sits at (0, 0), or when it is much further from
    its country's median location than the bulk of that country's points are.
    Large countries get a generous allowance because their points really are far apart.
    """
    bad = (df["lat_deg"].abs() < 1e-6) & (df["lon_deg"].abs() < 1e-6)
    bad |= ~df["lat_deg"].between(-90, 90) | ~df["lon_deg"].between(-180, 180)
    med = df.groupby("country_name")[["lat_deg", "lon_deg"]].transform("median")
    dist = pd.Series(
        haversine_km(df["lat_deg"], df["lon_deg"], med["lat_deg"], med["lon_deg"]),
        index=df.index,
    )
    q95 = dist.groupby(df["country_name"]).transform(lambda s: s.quantile(0.95))
    limit = np.maximum(3 * q95, 300.0)
    return bad | (dist > limit)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep rows with a usable label and sane geography. Returns data plus a tally."""
    tally = {"raw_rows": int(len(df))}
    out = df.copy()

    out = out[out["status_id"].isin(LABELLED)]
    tally["after_label_filter"] = int(len(out))

    dedupe_cols = [c for c in out.columns if c != "row_id"]
    out = out.drop_duplicates(subset=dedupe_cols)
    tally["after_duplicates"] = int(len(out))

    out = out.dropna(subset=["lat_deg", "lon_deg", "country_name", "report_date"])
    bad = flag_bad_coordinates(out)
    tally["bad_coordinates"] = int(bad.sum())
    out = out[~bad].copy()

    out["report_dt"] = pd.to_datetime(out["report_date"], errors="coerce", format="mixed")
    out = out.dropna(subset=["report_dt"])
    out["report_year"] = out["report_dt"].dt.year.astype("int32")
    window = out["report_year"].between(config.MIN_REPORT_YEAR, config.MAX_REPORT_YEAR)
    tally["report_date_outside_window"] = int((~window).sum())
    out = out[window].copy()
    out["report_month"] = out["report_dt"].dt.month.astype("int32")

    yr = out["install_year"]
    invalid = (yr < 1950) | (yr > out["report_year"])
    tally["invalid_install_year"] = int(invalid.sum())
    out.loc[invalid, "install_year"] = np.nan

    out["no_water"] = out["status_id"].map(LABELLED).astype("int8")
    out = out.reset_index(drop=True)
    tally["final_rows"] = int(len(out))
    tally["no_water_rate"] = float(out["no_water"].mean())
    tally["countries"] = int(out["country_name"].nunique())
    return out, tally


def make_sample(df: pd.DataFrame, n: int = 6000, seed: int = config.SEED) -> pd.DataFrame:
    """A small stratified-by-country sample used by tests, CI and the quick demo."""
    parts = []
    share = n / len(df)
    for _, grp in df.groupby("country_name"):
        k = max(20, int(round(len(grp) * share)))
        parts.append(grp.sample(min(k, len(grp)), random_state=seed))
    keep = [
        "row_id",
        "lat_deg",
        "lon_deg",
        "report_date",
        "status_id",
        "water_source",
        "water_tech",
        "facility_type",
        "country_name",
        "install_year",
        "installer",
        "pay",
        "status",
    ]
    return pd.concat(parts)[keep].sort_values("row_id")
