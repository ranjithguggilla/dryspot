"""Feature engineering. Everything learned from data lives inside the sklearn pipeline."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

CATEGORICAL = [
    "country_name",
    "water_source",
    "water_tech",
    "tech_family",
    "facility_type",
    "pay_class",
    "installer_clean",
]
NUMERIC_BASE = ["install_year", "age_at_report", "report_year", "report_month", "has_install_year"]
COORDS = ["lat_deg", "lon_deg"]
RAW_INPUT = [
    "lat_deg",
    "lon_deg",
    "report_date",
    "water_source",
    "water_tech",
    "facility_type",
    "country_name",
    "install_year",
    "installer",
    "pay",
]


def tech_family(value: object) -> str:
    """Collapse the many hand-pump brand strings into a handful of families."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    v = value.lower()
    if "hand pump" in v or "handpump" in v:
        return "hand pump"
    if "mechan" in v or "motor" in v or "solar" in v:
        return "mechanised pump"
    if "tap" in v or "kiosk" in v or "standpipe" in v:
        return "tapstand"
    if "rope" in v or "bucket" in v:
        return "rope and bucket"
    return "other"


_FREE = re.compile(r"\b(no|free|never|not pay|do not pay|don't pay)\b", re.I)
_FEE = re.compile(r"(fee|per |monthly|collect|yes|pay)", re.I)


def pay_class(value: object) -> str:
    """Free, fee-based or unknown. The raw text is far too messy to use directly."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    if _FREE.search(value):
        return "free"
    if _FEE.search(value):
        return "fee"
    return "other"


def clean_installer(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    v = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
    v = re.sub(r"\s+", " ", v).strip()
    return v or "unknown"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature construction from raw columns. Safe to apply to any single row."""
    out = pd.DataFrame(index=df.index)
    out["lat_deg"] = pd.to_numeric(df["lat_deg"], errors="coerce")
    out["lon_deg"] = pd.to_numeric(df["lon_deg"], errors="coerce")
    when = pd.to_datetime(df["report_date"], errors="coerce", format="mixed")
    out["report_year"] = when.dt.year.astype("float64")
    out["report_month"] = when.dt.month.astype("float64")
    install = pd.to_numeric(df["install_year"], errors="coerce")
    install = install.where((install >= 1950) & (install <= out["report_year"]))
    out["install_year"] = install
    out["has_install_year"] = install.notna().astype("float64")
    out["age_at_report"] = out["report_year"] - install
    out["country_name"] = df["country_name"].fillna("unknown").astype(str)
    for col in ("water_source", "water_tech", "facility_type"):
        out[col] = df[col].fillna("unknown").astype(str)
    out["tech_family"] = df["water_tech"].map(tech_family)
    out["pay_class"] = df["pay"].map(pay_class)
    out["installer_clean"] = df["installer"].map(clean_installer)
    return out


class NeighbourRate(BaseEstimator, TransformerMixin):
    """Share of nearby water points that were dry, computed without peeking.

    During fit_transform each training row is scored from its neighbours only
    (itself removed), so the feature has the same meaning at fit and predict time
    for a point that sits among surveyed neighbours. It does not fix the mismatch
    for points far from any training data, which is why the validation study
    scores it on held-out map blocks as well.
    """

    def __init__(self, k: int = config.NEIGHBOURS, prior_weight: float = 3.0):
        self.k = k
        self.prior_weight = prior_weight

    @staticmethod
    def _xyz(X) -> np.ndarray:
        """Unit-sphere coordinates, so a straight-line k-d tree gives great-circle neighbours."""
        arr = np.radians(np.nan_to_num(np.asarray(X, dtype="float64")[:, :2]))
        lat, lon = arr[:, 0], arr[:, 1]
        return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    def _query(self, X, k):
        chord, ind = self.tree_.query(self._xyz(X), k=k, workers=-1)
        arc_km = 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * config.EARTH_RADIUS_KM
        return arc_km, ind

    def fit(self, X, y):
        self.y_ = np.asarray(y, dtype="float64")
        self.prior_ = float(self.y_.mean())
        self.tree_ = cKDTree(self._xyz(X))
        return self

    def _summarise(self, dist, ind):
        rate = (self.y_[ind].sum(axis=1) + self.prior_weight * self.prior_) / (
            ind.shape[1] + self.prior_weight
        )
        km = dist
        return np.column_stack([rate, np.median(km, axis=1), km.max(axis=1)])

    def transform(self, X):
        dist, ind = self._query(X, self.k)
        return self._summarise(dist, ind)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        n = len(self.y_)
        dist, ind = self._query(X, self.k + 1)
        self_hit = ind == np.arange(n)[:, None]
        no_self = ~self_hit.any(axis=1)
        self_hit[no_self, -1] = True  # duplicates at one spot: drop the farthest instead
        keep = ~self_hit
        return self._summarise(dist[keep].reshape(n, self.k), ind[keep].reshape(n, self.k))

    def get_feature_names_out(self, input_features=None):
        return np.array(["nn_dry_rate", "nn_median_km", "nn_max_km"])


def build_preprocessor(
    use_coords: bool, use_neighbours: bool, linear: bool, drop: tuple[str, ...] = ()
) -> ColumnTransformer:
    """Column transformer for the engineered frame. `linear` picks one-hot plus scaling."""
    num = [c for c in NUMERIC_BASE if c not in drop]
    if linear:
        cat_enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200)
        num_enc = Pipeline(
            [
                ("impute", _median_imputer()),
                ("scale", StandardScaler()),
            ]
        )
    else:
        cat_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            min_frequency=200,
            max_categories=120,
        )
        num_enc = "passthrough"
    parts = [("cat", cat_enc, CATEGORICAL), ("num", num_enc, num)]
    if use_coords and not linear:
        parts.append(("coords", "passthrough", COORDS))
    if use_neighbours:
        parts.append(("nn", NeighbourRate(), COORDS))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.3)


def _median_imputer():
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy="median", keep_empty_features=True)
