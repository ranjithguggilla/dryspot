"""Charts for the README. One function per figure, each writes a png."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1f2933"
BLUE = "#1f6f8b"
SAND = "#d9a441"
RED = "#b8483c"
GREY = "#9aa5b1"
PALETTE = [GREY, "#8fb3d1", "#4f7ea8", BLUE, SAND, RED, "#4a3f6b"]


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def validation_gap(results: pd.DataFrame, path: Path):
    """AUC for each model under each way of holding data out."""
    regimes = ["random", "spatial_block", "country_holdout", "forward_in_time"]
    labels = ["random rows", "map blocks", "whole countries", "later surveys"]
    models = list(results["model"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [
            results[(results.model == m) & (results.regime == r)]["auc"].iloc[0] for r in regimes
        ]
        ax.bar(
            np.arange(len(regimes)) + i * width,
            vals,
            width,
            label=m.replace("_", " "),
            color=PALETTE[i % len(PALETTE)],
        )
    ax.set_xticks(np.arange(len(regimes)) + width * (len(models) - 1) / 2, labels)
    ax.set_ylim(0.45, 0.95)
    ax.axhline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_ylabel("ROC AUC on held-out rows")
    ax.set_title("The same models, scored four ways")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    _style(ax)
    _save(fig, path)


def dry_map(df: pd.DataFrame, path: Path):
    """Observed dry share per map block, drawn at the block's mean position."""
    g = df.groupby("block").agg(
        lat=("lat_deg", "mean"),
        lon=("lon_deg", "mean"),
        n=("no_water", "size"),
        dry=("no_water", "mean"),
    )
    g = g[g.n >= 30]
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        g.lon,
        g.lat,
        c=g.dry,
        s=np.clip(g.n / 25, 6, 80),
        cmap="YlOrRd",
        vmin=0,
        vmax=0.8,
        edgecolor="none",
        alpha=0.9,
    )
    ax.set_xlim(-20, 52)
    ax.set_ylim(-36, 38)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Share of water points with no water at the visit, by map block")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="dry share")
    _style(ax)
    _save(fig, path)


def calibration(table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=":", label="perfect")
    ax.plot(table.predicted, table.observed, "o-", color=BLUE, label="shipped model")
    ax.set_xlabel("predicted chance of no water")
    ax.set_ylabel("observed share with no water")
    ax.set_title("Reliability on held-out map blocks")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, path)


def precision_curve(sweep: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = sweep.share_visited * 100
    ax.plot(x, sweep.model * 100, "o-", color=BLUE, label="model ranking")
    ax.plot(x, sweep.country_prior * 100, "o-", color=SAND, label="country prior only")
    ax.plot(x, sweep.random * 100, "--", color=GREY, label="random visits")
    ax.set_xlabel("share of water points visited (%)")
    ax.set_ylabel("visits that find a dry point (%)")
    ax.set_title("Hit rate of a visit budget, held-out map blocks")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, path)


def importance(imp: pd.DataFrame, path: Path):
    imp = imp.sort_values("auc_drop")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(imp.group, imp.auc_drop, xerr=imp.auc_drop_sd, color=BLUE, ecolor=INK)
    ax.set_xlabel("AUC lost when the group is shuffled")
    ax.set_title("What the shipped model leans on")
    _style(ax)
    _save(fig, path)


def country_auc(table: pd.DataFrame, path: Path):
    t = table.sort_values("auc")
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.barh(t.country, t.auc, color=[RED if a < 0.6 else BLUE for a in t.auc])
    ax.axvline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("ROC AUC when the country is entirely unseen")
    ax.set_title("Country hold-out, per country")
    _style(ax)
    _save(fig, path)
