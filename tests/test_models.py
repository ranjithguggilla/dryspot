import numpy as np
import pytest

from dryspot import models as m


@pytest.mark.parametrize("name", m.MODEL_NAMES)
def test_every_model_fits_and_gives_probabilities(name, cleaned):
    sub = cleaned.iloc[:3000]
    model = m.make_model(name).fit(sub, sub["no_water"])
    p = m.predict_dry(model, cleaned.iloc[3000:3300])
    assert p.shape == (300,)
    assert ((p >= 0) & (p <= 1)).all()


def test_country_rate_matches_hand_computation(cleaned):
    sub = cleaned[cleaned["country_name"].isin(["Uganda"])]
    base = m.CountryRate(prior_weight=0.0).fit(sub, sub["no_water"])
    assert np.isclose(base.rates_["Uganda"], sub["no_water"].mean())


def test_country_rate_falls_back_to_prior_for_new_country(cleaned):
    base = m.CountryRate().fit(cleaned, cleaned["no_water"])
    probe = cleaned.iloc[:2].copy()
    probe["country_name"] = "Atlantis"
    assert np.allclose(m.predict_dry(base, probe), base.prior_)


def test_boosted_model_beats_country_rate_on_random_split(cleaned):
    n = len(cleaned)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: n * 3 // 4], idx[n * 3 // 4 :]
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for name in ("country_rate", "boosted_with_location"):
        model = m.make_model(name).fit(cleaned.iloc[tr], cleaned["no_water"].iloc[tr])
        aucs[name] = roc_auc_score(
            cleaned["no_water"].iloc[te], m.predict_dry(model, cleaned.iloc[te])
        )
    assert aucs["boosted_with_location"] > aucs["country_rate"]


def test_unknown_model_name():
    with pytest.raises(KeyError):
        m.make_model("magic")


def test_model_scores_a_single_raw_row(cleaned):
    sub = cleaned.iloc[:2000]
    model = m.make_model("boosted_with_location").fit(sub, sub["no_water"])
    one = cleaned.iloc[[2500]]
    assert m.predict_dry(model, one).shape == (1,)
