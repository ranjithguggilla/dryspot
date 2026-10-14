"""Plain-language reasons for a score, and grouped permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .models import predict_dry

GROUPS: dict[str, list[str]] = {
    "location": ["lat_deg", "lon_deg"],
    "country": ["country_name"],
    "water source": ["water_source"],
    "pump technology": ["water_tech"],
    "facility type": ["facility_type"],
    "installation year": ["install_year"],
    "survey timing": ["report_date"],
    "installer": ["installer"],
    "payment arrangement": ["pay"],
}


def permutation_importance(
    model, X: pd.DataFrame, y, repeats: int = 3, seed: int = 0
) -> pd.DataFrame:
    """AUC lost when a group of raw columns is shuffled together on held-out rows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = roc_auc_score(y, predict_dry(model, X))
    rows = []
    for name, cols in GROUPS.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            perm = rng.permutation(len(X))
            for c in cols:
                shuffled[c] = X[c].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, predict_dry(model, shuffled)))
        rows.append((name, float(np.mean(drops)), float(np.std(drops))))
    out = pd.DataFrame(rows, columns=["group", "auc_drop", "auc_drop_sd"])
    return out.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def reasons(
    model,
    row: pd.DataFrame,
    background: pd.DataFrame,
    n_draws: int = 60,
    seed: int = 0,
    top: int = 3,
) -> list[dict]:
    """How much each group of facts moves this point's risk versus a typical alternative.

    For each group the row's values are swapped for values drawn from real water
    points and the predicted risk is averaged. Positive means the actual value
    makes this point riskier than the typical alternative. Swaps can create
    unusual combinations, so treat the numbers as a guide and not as causes.
    """
    rng = np.random.default_rng(seed)
    p_actual = float(predict_dry(model, row)[0])
    draws = background.iloc[rng.integers(0, len(background), n_draws)].reset_index(drop=True)
    out = []
    for name, cols in GROUPS.items():
        trial = pd.concat([row] * n_draws, ignore_index=True)
        for c in cols:
            trial[c] = draws[c].to_numpy()
        p_typical = float(predict_dry(model, trial).mean())
        out.append({"group": name, "effect": p_actual - p_typical, "p_typical": p_typical})
    out.sort(key=lambda r: abs(r["effect"]), reverse=True)
    return out[:top]


def sentence(item: dict, row: pd.DataFrame) -> str:
    r = row.iloc[0].copy()
    for key in ("install_year", "installer", "pay", "water_tech", "water_source", "facility_type"):
        if pd.isna(r[key]) or str(r[key]).strip() == "":
            r[key] = "not recorded"
    if str(r["install_year"]) != "not recorded":
        r["install_year"] = int(float(r["install_year"]))
    detail = {
        "location": "its position on the map",
        "country": f"being in {r['country_name']}",
        "water source": f"the source type ({r['water_source']})",
        "pump technology": f"the technology ({r['water_tech']})",
        "facility type": f"the facility class ({r['facility_type']})",
        "installation year": f"the installation year ({r['install_year']})",
        "survey timing": f"when it was surveyed ({r['report_date']})",
        "installer": f"the installer ({r['installer']})",
        "payment arrangement": f"the payment arrangement ({r['pay']})",
    }[item["group"]]
    direction = "raises" if item["effect"] > 0 else "lowers"
    points = abs(item["effect"]) * 100
    return f"{detail[0].upper()}{detail[1:]} {direction} the risk by about {points:.1f} points."
