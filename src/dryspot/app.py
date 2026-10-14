"""Gradio app: check one water point or rank a whole list."""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from . import config, mapview, score

PAY_CHOICES = [
    "",
    "No payment, free",
    "Water committee collects fees",
    "Pay per bucket",
    "Monthly fee",
]


def check_point(
    bundle: dict, country, lat, lon, source, tech, facility, install_year, installer, pay
):
    """Function behind the single-point tab. Returns markdown, kept free of gradio types."""
    rec = {
        "country_name": country,
        "lat_deg": lat,
        "lon_deg": lon,
        "water_source": source or None,
        "water_tech": tech or None,
        "facility_type": facility or None,
        "install_year": install_year if install_year else None,
        "installer": installer or None,
        "pay": pay or None,
    }
    res = score.score_one(bundle, rec)
    lines = [f"### Chance of no water at the next visit: {res['risk']:.0%} ({res['band']})", ""]
    lines += [f"- {r}" for r in res["reasons"]]
    return "\n".join(lines)


def rank_list(bundle: dict, file_path: str | None, share: float):
    """Function behind the batch tab. Returns a table, a csv path and map html."""
    if not file_path:
        return pd.DataFrame(), None, "<p>Upload a csv first.</p>"
    df = pd.read_csv(file_path)
    needed = {"lat_deg", "lon_deg", "country_name"}
    missing = needed - set(df.columns)
    if missing:
        return pd.DataFrame({"problem": [f"missing columns: {sorted(missing)}"]}), None, ""
    scored = score.triage_split(score.score_frame(bundle, df), share)
    out_path = Path(tempfile.mkdtemp()) / "ranked_water_points.csv"
    scored.to_csv(out_path, index=False)
    fmap = mapview.risk_map(scored)
    doc = html.escape(fmap.get_root().render())
    frame = f'<iframe srcdoc="{doc}" width="100%" height="480"></iframe>'
    return scored.head(200), str(out_path), frame


def build_app(bundle: dict) -> gr.Blocks:
    meta = bundle["meta"]
    with gr.Blocks(title="dryspot") as demo:
        gr.Markdown(
            "# dryspot\nChance that a water point has no water on the next visit. "
            "Scores come from a model checked on map areas it never saw."
        )
        with gr.Tab("One water point"):
            country = gr.Dropdown(meta["countries"], value="Uganda", label="Country")
            with gr.Row():
                lat = gr.Number(value=1.37, label="Latitude")
                lon = gr.Number(value=32.29, label="Longitude")
            source = gr.Dropdown(
                [""] + meta["water_sources"], value="Borehole", label="Water source"
            )
            tech = gr.Dropdown(
                [""] + meta["water_techs"], value="Hand Pump - Afridev", label="Technology"
            )
            facility = gr.Dropdown(
                [""] + meta["facility_types"], value="Improved", label="Facility type"
            )
            year = gr.Number(value=2008, precision=0, label="Install year (0 if unknown)")
            installer = gr.Textbox(label="Installer (optional)")
            pay = gr.Dropdown(PAY_CHOICES, value="", label="Payment arrangement")
            out = gr.Markdown()
            gr.Button("Score").click(
                lambda *a: check_point(bundle, *a),
                [country, lat, lon, source, tech, facility, year, installer, pay],
                out,
            )
        with gr.Tab("Rank a list"):
            gr.Markdown(
                "Upload a csv with `lat_deg`, `lon_deg`, `country_name` plus any other columns."
            )
            up = gr.File(file_types=[".csv"], type="filepath")
            share = gr.Slider(
                0.02, 0.5, value=config.DEFAULT_VISIT_SHARE, label="Share to visit first"
            )
            table = gr.Dataframe()
            dl = gr.File(label="Ranked csv")
            fmap = gr.HTML()
            gr.Button("Rank").click(
                lambda f, s: rank_list(bundle, f, s), [up, share], [table, dl, fmap]
            )
    return demo


def launch(model_path: str | None = None) -> None:
    build_app(score.load_bundle(model_path)).launch()
