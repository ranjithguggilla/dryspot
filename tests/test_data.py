import numpy as np
import pandas as pd

from dryspot import data


def make_raw(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "row_id": range(n),
            "lat_deg": rng.normal(1.0, 0.3, n),
            "lon_deg": rng.normal(32.0, 0.3, n),
            "report_date": ["03/15/2014 12:00:00 AM"] * n,
            "status_id": ["y", "n", "u", "y"] * (n // 4),
            "water_source": "Borehole",
            "water_tech": "Hand Pump",
            "facility_type": "Improved",
            "country_name": "Uganda",
            "install_year": 2005.0,
            "installer": "x",
            "pay": "No",
            "status": "ok",
        }
    )


def test_haversine_known_distance():
    # London to Paris is about 344 km
    d = data.haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 340 < float(d) < 348


def test_clean_keeps_only_labelled_rows_and_maps_target():
    df, tally = data.clean(make_raw())
    assert set(df["status_id"]) == {"y", "n"}
    assert df["no_water"].tolist() == [1 if s == "n" else 0 for s in df["status_id"]]
    assert tally["final_rows"] == len(df)


def test_clean_removes_far_away_coordinates():
    raw = make_raw()
    raw.loc[0, ["lat_deg", "lon_deg"]] = [-60.0, 120.0]  # labelled y, nowhere near Uganda
    raw.loc[1, ["lat_deg", "lon_deg"]] = [0.0, 0.0]
    raw.loc[0, "status_id"] = "y"
    raw.loc[1, "status_id"] = "n"
    df, tally = data.clean(raw)
    assert tally["bad_coordinates"] >= 2
    assert 0 not in set(df["row_id"]) and 1 not in set(df["row_id"])


def test_clean_nulls_impossible_install_years():
    raw = make_raw()
    raw.loc[0, "install_year"] = 2045.0
    raw.loc[0, "status_id"] = "y"
    raw.loc[4, "install_year"] = 1800.0
    raw.loc[4, "status_id"] = "y"
    df, tally = data.clean(raw)
    assert df.loc[df["row_id"] == 0, "install_year"].isna().all()
    assert tally["invalid_install_year"] >= 2


def test_clean_drops_duplicates_ignoring_row_id():
    raw = make_raw()
    dup = raw.iloc[[0]].copy()
    dup["row_id"] = 999
    raw.loc[0, "status_id"] = "y"
    dup["status_id"] = "y"
    df, tally = data.clean(pd.concat([raw, dup]))
    assert tally["after_duplicates"] == tally["after_label_filter"] - 1


def test_clean_drops_report_dates_outside_window():
    raw = make_raw()
    raw.loc[0, "report_date"] = "03/15/1899 12:00:00 AM"
    raw.loc[0, "status_id"] = "y"
    df, tally = data.clean(raw)
    assert tally["report_date_outside_window"] == 1
    assert 0 not in set(df["row_id"])


def test_sample_is_small_and_covers_countries(raw_sample):
    assert 4000 < len(raw_sample) < 9000
    assert raw_sample["country_name"].nunique() > 20
