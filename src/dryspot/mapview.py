"""Interactive map of a scored list, built with folium."""

from __future__ import annotations

import folium
import pandas as pd

COLOURS = {"low": "#2c7fb8", "moderate": "#f0b429", "high": "#e8590c", "very high": "#b02a37"}


def risk_map(scored: pd.DataFrame, max_points: int = 1500) -> folium.Map:
    """Circle markers coloured by risk band. Shows the highest-risk points if the list is long."""
    pts = scored.dropna(subset=["lat_deg", "lon_deg"]).nsmallest(max_points, "visit_rank")
    if pts.empty:
        return folium.Map(location=[5, 20], zoom_start=3, tiles="OpenStreetMap")
    centre = [pts["lat_deg"].mean(), pts["lon_deg"].mean()]
    fmap = folium.Map(location=centre, zoom_start=6, tiles="OpenStreetMap")
    for r in pts.itertuples():
        folium.CircleMarker(
            [r.lat_deg, r.lon_deg],
            radius=5,
            weight=1,
            color="#333",
            fill=True,
            fill_color=COLOURS.get(r.band, "#666"),
            fill_opacity=0.85,
            tooltip=f"rank {r.visit_rank}, risk {r.risk:.0%} ({r.band})",
        ).add_to(fmap)
    lo, hi = pts[["lat_deg", "lon_deg"]].min().tolist(), pts[["lat_deg", "lon_deg"]].max().tolist()
    fmap.fit_bounds([lo, hi])
    return fmap
