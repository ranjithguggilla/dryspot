import numpy as np
import pandas as pd
import pytest

from dryspot import features as f


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hand Pump - Afridev", "hand pump"),
        ("Mechanized Pump", "mechanised pump"),
        ("Tapstand", "tapstand"),
        ("Rope and Bucket", "rope and bucket"),
        ("Something odd", "other"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_tech_family(text, expected):
    assert f.tech_family(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("No payment – its free", "free"),
        ("Never pay", "free"),
        ("They do not pay", "free"),
        ("Water Committee Collects Fees", "fee"),
        ("Pay per bucket", "fee"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_pay_class(text, expected):
    assert f.pay_class(text) == expected


def test_clean_installer_normalises_case_and_punctuation():
    assert f.clean_installer("  W.H.H. / Gov't ") == "w h h gov t"
    assert f.clean_installer(None) == "unknown"


def raw_row(**kw):
    base = {
        "lat_deg": 1.0,
        "lon_deg": 32.0,
        "report_date": "2014-03-15",
        "water_source": "Borehole",
        "water_tech": "Hand Pump",
        "facility_type": "Improved",
        "country_name": "Uganda",
        "install_year": 2004.0,
        "installer": "DWD",
        "pay": "No",
    }
    base.update(kw)
    return pd.DataFrame([base])


def test_engineer_single_row_is_self_contained():
    out = f.engineer(raw_row())
    assert out.loc[0, "age_at_report"] == 10
    assert out.loc[0, "has_install_year"] == 1.0
    assert out.loc[0, "tech_family"] == "hand pump"


def test_engineer_handles_missing_values():
    out = f.engineer(raw_row(install_year=np.nan, installer=None, pay=None, water_tech=None))
    assert np.isnan(out.loc[0, "age_at_report"])
    assert out.loc[0, "has_install_year"] == 0.0
    assert out.loc[0, "pay_class"] == "unknown"
    assert out.loc[0, "installer_clean"] == "unknown"


def test_engineer_rejects_install_year_after_report():
    out = f.engineer(raw_row(install_year=2030.0))
    assert np.isnan(out.loc[0, "install_year"])


def test_neighbour_rate_excludes_the_row_itself():
    # four points on a line; with k=1 each row's nearest other point is unambiguous
    X = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 5.0], [0.0, 5.1]])
    y = np.array([1, 0, 1, 1])
    nr = f.NeighbourRate(k=1, prior_weight=0.0)
    out = nr.fit_transform(X, y)
    assert out[:, 0].tolist() == [0.0, 1.0, 1.0, 1.0]  # each takes its partner's label


def test_neighbour_rate_handles_duplicate_coordinates():
    X = np.array([[1.0, 1.0]] * 5 + [[2.0, 2.0]])
    y = np.array([1, 0, 1, 0, 1, 0])
    out = f.NeighbourRate(k=2, prior_weight=0.0).fit_transform(X, y)
    assert out.shape == (6, 3) and np.isfinite(out).all()


def test_neighbour_rate_transform_differs_from_fit_transform():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200, 2))
    y = (X[:, 0] > 0.5).astype(int)
    nr = f.NeighbourRate(k=5)
    train_feat = nr.fit_transform(X, y)
    again = nr.transform(X)
    # transform sees the row itself as its own neighbour, fit_transform must not
    assert not np.allclose(train_feat[:, 0], again[:, 0])


def test_neighbour_distance_is_in_kilometres():
    X = np.array([[0.0, 0.0], [0.0, 1.0]])  # one degree of longitude at the equator, 111.2 km
    nr = f.NeighbourRate(k=1).fit(X, np.array([0, 1]))
    km = nr._query(np.array([[0.0, 0.0]]), 2)[0][0, 1]
    assert 110 < km < 112.5
