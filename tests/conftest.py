from __future__ import annotations

import pytest

from dryspot import config, data, study, validation


@pytest.fixture(scope="session")
def raw_sample():
    return data.load_raw(config.SAMPLE_FILE)


@pytest.fixture(scope="session")
def cleaned(raw_sample):
    df, _ = data.clean(raw_sample)
    return validation.add_cells(df)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, cleaned):
    """A small end-to-end study on the sample, shared by several test modules."""
    out = tmp_path_factory.mktemp("out")
    assets = tmp_path_factory.mktemp("assets")
    metrics = study.run_study(
        cleaned,
        out,
        assets,
        n_splits=3,
        n_boot=20,
        models=[
            "country_rate",
            "logistic",
            "boosted_attributes",
            "boosted_location_no_timing",
            "boosted_with_neighbours",
        ],
    )
    return {"metrics": metrics, "out": out, "assets": assets}
