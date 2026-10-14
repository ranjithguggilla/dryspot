"""What a ranking is worth to a field team, with every assumption in the open."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .evaluate import block_bootstrap, paired_block_bootstrap, precision_at_share


def dispatch_table(
    y,
    p_model,
    p_baseline,
    groups,
    share: float = config.DEFAULT_VISIT_SHARE,
    cost_per_visit: float = config.COST_PER_VISIT_USD,
    visits_per_day: int = config.VISITS_PER_TECH_DAY,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Compare three ways to spend a fixed visit budget: random, country prior, model."""
    y = np.asarray(y)
    stat = lambda yy, pp: precision_at_share(yy, pp, share)  # noqa: E731
    prevalence = float(y.mean())
    m, m_lo, m_hi = block_bootstrap(y, p_model, groups, stat, n_boot, seed)
    c, c_lo, c_hi = block_bootstrap(y, p_baseline, groups, stat, n_boot, seed)
    d, d_lo, d_hi = paired_block_bootstrap(y, p_model, p_baseline, groups, stat, n_boot, seed)

    def row(prec):
        return {
            "faults_per_1000_visits": 1000 * prec,
            "cost_per_fault_usd": cost_per_visit / prec,
            "tech_days_per_100_faults": 100 / prec / visits_per_day,
        }

    out = {
        "share_visited": share,
        "assumptions": {
            "cost_per_visit_usd": cost_per_visit,
            "visits_per_tech_day": visits_per_day,
            "note": "A visit confirms and logs a fault; repair cost is excluded.",
        },
        "random": {"precision": prevalence, **row(prevalence)},
        "country_prior": {"precision": c, "precision_ci": [c_lo, c_hi], **row(c)},
        "model": {"precision": m, "precision_ci": [m_lo, m_hi], **row(m)},
        "model_minus_country_prior": {"precision_gain": d, "ci": [d_lo, d_hi]},
    }
    out["saving_per_1000_faults_vs_random_usd"] = (1000 / prevalence - 1000 / m) * cost_per_visit
    out["saving_per_1000_faults_vs_country_prior_usd"] = (1000 / c - 1000 / m) * cost_per_visit
    return out


def share_sweep(y, p_model, p_baseline, shares=(0.02, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    y = np.asarray(y)
    rows = []
    for s in shares:
        rows.append(
            {
                "share_visited": s,
                "random": float(y.mean()),
                "country_prior": precision_at_share(y, p_baseline, s),
                "model": precision_at_share(y, p_model, s),
            }
        )
    return pd.DataFrame(rows)
