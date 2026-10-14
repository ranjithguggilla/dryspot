"""Paths, constants and defaults shared across the package."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FILE = RAW_DIR / "water.csv"
SAMPLE_FILE = DATA_DIR / "sample.csv.gz"
OUTPUT_DIR = ROOT / "outputs"
ASSET_DIR = ROOT / "assets"

RAW_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/"
    "data/2021/2021-05-04/water.csv"
)

SEED = 17

# h3 resolution used to cut the map into validation blocks (about 60 km edge)
BLOCK_RESOLUTION = 3
# finer cells used only for reporting and for grouping bootstrap resamples
CELL_RESOLUTION = 5

NEIGHBOURS = 10
EARTH_RADIUS_KM = 6371.0088

MIN_REPORT_YEAR = 2000
MAX_REPORT_YEAR = 2021

# year split for the forward-in-time check
SPLIT_YEAR = 2015

# economics: every figure is an assumption the reader can change
COST_PER_VISIT_USD = 35.0
VISITS_PER_TECH_DAY = 6
DEFAULT_VISIT_SHARE = 0.10
