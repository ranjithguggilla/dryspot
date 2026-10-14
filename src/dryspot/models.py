"""Candidate models. Each one is a full pipeline, raw frame in and probability out."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import config
from .features import CATEGORICAL, build_preprocessor, engineer


class CountryRate(BaseEstimator, ClassifierMixin):
    """Baseline: predict each country's historical dry rate. Hard to beat cheaply."""

    def __init__(self, prior_weight: float = 20.0):
        self.prior_weight = prior_weight

    def fit(self, X, y):
        y = np.asarray(y, dtype="float64")
        self.prior_ = float(y.mean())
        grp = pd.DataFrame({"c": X["country_name"].to_numpy(), "y": y}).groupby("c")["y"]
        n, s = grp.size(), grp.sum()
        self.rates_ = ((s + self.prior_weight * self.prior_) / (n + self.prior_weight)).to_dict()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["country_name"].map(self.rates_).fillna(self.prior_).to_numpy(dtype="float64")
        return np.column_stack([1 - p, p])


def _wrap(preproc, estimator) -> Pipeline:
    return Pipeline(
        [
            ("engineer", FunctionTransformer(engineer)),
            ("prep", preproc),
            ("clf", estimator),
        ]
    )


def _booster(n_categorical: int, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=48,
        min_samples_leaf=60,
        l2_regularization=1.0,
        categorical_features=list(range(n_categorical)),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=seed,
    )


MODEL_NAMES = [
    "country_rate",
    "logistic",
    "boosted_attributes",
    "boosted_location_no_timing",
    "boosted_location_no_year",
    "boosted_with_location",
    "boosted_with_neighbours",
]


def make_model(name: str, seed: int = config.SEED):
    """Build an unfitted model by name."""
    if name == "country_rate":
        return Pipeline([("engineer", FunctionTransformer(engineer)), ("clf", CountryRate())])
    if name == "logistic":
        clf = LogisticRegression(C=0.5, max_iter=300, solver="lbfgs")
        return _wrap(build_preprocessor(False, False, linear=True), clf)
    if name == "boosted_attributes":
        return _wrap(
            build_preprocessor(False, False, linear=False), _booster(len(CATEGORICAL), seed)
        )
    if name == "boosted_location_no_timing":
        prep = build_preprocessor(True, False, linear=False, drop=("report_year", "report_month"))
        return _wrap(prep, _booster(len(CATEGORICAL), seed))
    if name == "boosted_location_no_year":
        prep = build_preprocessor(True, False, linear=False, drop=("report_year",))
        return _wrap(prep, _booster(len(CATEGORICAL), seed))
    if name == "boosted_with_location":
        return _wrap(
            build_preprocessor(True, False, linear=False), _booster(len(CATEGORICAL), seed)
        )
    if name == "boosted_with_neighbours":
        return _wrap(build_preprocessor(True, True, linear=False), _booster(len(CATEGORICAL), seed))
    raise KeyError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def predict_dry(model, X: pd.DataFrame) -> np.ndarray:
    """Probability that the water point has no water on the visit."""
    return model.predict_proba(X)[:, 1]
