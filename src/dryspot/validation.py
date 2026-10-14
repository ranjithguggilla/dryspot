"""Four ways to hold data out. They answer four different business questions."""

from __future__ import annotations

from collections.abc import Iterator

import h3
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from . import config


def add_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Attach h3 block and cell ids used for splitting and for bootstrap grouping."""
    out = df.copy()
    lat, lon = out["lat_deg"].to_numpy(), out["lon_deg"].to_numpy()
    pairs = list(zip(lat, lon, strict=True))
    out["block"] = [h3.latlng_to_cell(a, b, config.BLOCK_RESOLUTION) for a, b in pairs]
    out["cell"] = [h3.latlng_to_cell(a, b, config.CELL_RESOLUTION) for a, b in pairs]
    return out


def random_folds(
    df: pd.DataFrame, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Plain stratified shuffle. Neighbours of a test point sit in the training set."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(df, df["no_water"])


def spatial_folds(
    df: pd.DataFrame, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Whole map blocks held out together: 'how will this do in a place it has not seen'."""
    blocks = df["block"].to_numpy()
    uniq = pd.unique(blocks)
    rng = np.random.default_rng(seed)
    order = {b: i for i, b in enumerate(rng.permutation(uniq))}
    groups = np.fromiter((order[b] for b in blocks), dtype="int64", count=len(blocks))
    yield from GroupKFold(n_splits=n_splits).split(df, groups=groups)


def country_folds(
    df: pd.DataFrame, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Whole countries held out: 'what if we launch in a new country'."""
    yield from GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        df, groups=df["country_name"].to_numpy()
    )


def forward_split(df: pd.DataFrame, split_year: int = config.SPLIT_YEAR):
    """Train on early survey rounds, test on later ones. Returns a single fold."""
    years = df["report_year"].to_numpy()
    train = np.flatnonzero(years <= split_year)
    test = np.flatnonzero(years > split_year)
    yield train, test


REGIMES = {
    "random": random_folds,
    "spatial_block": spatial_folds,
    "country_holdout": country_folds,
}


def folds_for(regime: str, df: pd.DataFrame, n_splits: int, seed: int):
    if regime == "forward_in_time":
        return list(forward_split(df))
    if regime not in REGIMES:
        raise KeyError(f"unknown regime {regime!r}")
    return list(REGIMES[regime](df, n_splits, seed))
