import numpy as np

from dryspot import validation as v


def test_spatial_folds_never_share_a_block(cleaned):
    for tr, te in v.folds_for("spatial_block", cleaned, 3, 1):
        assert not set(cleaned["block"].iloc[tr]) & set(cleaned["block"].iloc[te])


def test_country_folds_never_share_a_country(cleaned):
    for tr, te in v.folds_for("country_holdout", cleaned, 3, 1):
        assert not set(cleaned["country_name"].iloc[tr]) & set(cleaned["country_name"].iloc[te])


def test_forward_split_respects_time(cleaned):
    ((tr, te),) = v.folds_for("forward_in_time", cleaned, 3, 1)
    assert cleaned["report_year"].iloc[tr].max() <= 2015 < cleaned["report_year"].iloc[te].min()


def test_random_folds_cover_every_row_once(cleaned):
    seen = np.concatenate([te for _, te in v.folds_for("random", cleaned, 4, 1)])
    assert sorted(seen) == list(range(len(cleaned)))


def test_random_split_leaks_neighbours_but_spatial_does_not(cleaned):
    def shared_share(tr, te):
        cells_tr = set(cleaned["cell"].iloc[tr])
        return float(cleaned["cell"].iloc[te].isin(cells_tr).mean())

    tr, te = v.folds_for("random", cleaned, 4, 1)[0]
    assert shared_share(tr, te) > 0.5
    tr, te = v.folds_for("spatial_block", cleaned, 4, 1)[0]
    # h3 cells at different resolutions do not nest perfectly, so allow a sliver at block edges
    assert shared_share(tr, te) < 0.05


def test_unknown_regime_is_rejected(cleaned):
    import pytest

    with pytest.raises(KeyError):
        v.folds_for("nonsense", cleaned, 3, 1)
