import joblib
import numpy as np
import pandas as pd
import pytest

from dryspot import app, cli, explain, mapview, score, study


@pytest.fixture(scope="module")
def bundle(trained):
    return joblib.load(trained["out"] / "model.joblib")


@pytest.fixture(scope="module")
def sample_rows(raw_sample):
    return raw_sample.sample(60, random_state=3).drop(columns=["status_id", "status"])


def test_study_wrote_every_artefact(trained):
    out, assets = trained["out"], trained["assets"]
    for name in (
        "validation_results.csv",
        "model.joblib",
        "oof_predictions.parquet",
        "permutation_importance.csv",
        "precision_sweep.csv",
    ):
        assert (out / name).exists(), name
    for name in (
        "validation_gap.png",
        "dry_map.png",
        "calibration.png",
        "precision_curve.png",
        "importance.png",
        "country_holdout.png",
    ):
        assert (assets / name).exists(), name


def test_random_validation_is_more_optimistic_for_the_neighbour_model(trained):
    res = pd.DataFrame(trained["metrics"]["validation"])
    pick = res[res.model == "boosted_with_neighbours"].set_index("regime")["auc"]
    assert pick["random"] > pick["spatial_block"]


def test_metrics_have_expected_sections(trained):
    m = trained["metrics"]
    for key in (
        "selection",
        "calibration",
        "conformal",
        "economics",
        "status_text_audit",
        "importance",
    ):
        assert key in m
    assert 0.5 < m["shipped_spatial_auc"]["auc"] < 1.0


def test_choose_model_takes_the_simplest_one_that_ties_the_best():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.integers(0, 2, n)
    cells = rng.integers(0, 200, n)
    signal = y * 0.7 + rng.normal(0, 1, n)
    oof = {
        "country_rate": rng.uniform(size=n),
        "logistic": signal,
        "boosted_attributes": signal + rng.normal(0, 0.01, n),  # same skill, more complex
    }
    chosen, detail = study.choose_model(y, oof, cells, n_boot=60)
    assert chosen == "logistic"
    assert detail["gap_to_best"]["country_rate"]["ci"][0] > 0


def test_choose_model_keeps_the_best_when_it_clearly_wins():
    rng = np.random.default_rng(1)
    n = 20000
    y = rng.integers(0, 2, n)
    cells = rng.integers(0, 200, n)
    oof = {
        "logistic": y * 0.2 + rng.normal(0, 1, n),
        "boosted_with_location": y * 1.5 + rng.normal(0, 1, n),
    }
    chosen, _ = study.choose_model(y, oof, cells, n_boot=60)
    assert chosen == "boosted_with_location"


def test_score_frame_ranks_highest_risk_first(bundle, sample_rows):
    out = score.score_frame(bundle, sample_rows)
    assert out["visit_rank"].tolist() == list(range(1, len(out) + 1))
    assert out["risk"].is_monotonic_decreasing
    assert out["risk"].between(0, 1).all()
    assert set(out["band"]) <= {"low", "moderate", "high", "very high"}


def test_score_frame_survives_missing_columns(bundle):
    thin = pd.DataFrame(
        {"lat_deg": [1.0, 2.0], "lon_deg": [32.0, 33.0], "country_name": ["Uganda"] * 2}
    )
    out = score.score_frame(bundle, thin)
    assert len(out) == 2 and out["risk"].notna().all()


def test_band_edges():
    assert score.band(0.0) == "low" and score.band(0.15) == "moderate"
    assert score.band(0.35) == "high" and score.band(0.99) == "very high"


def test_score_one_returns_reasons(bundle):
    res = score.score_one(
        bundle,
        {
            "country_name": "Uganda",
            "lat_deg": 1.4,
            "lon_deg": 32.3,
            "water_tech": "Hand Pump - Afridev",
            "install_year": 2001,
        },
    )
    assert 0 <= res["risk"] <= 1 and len(res["reasons"]) == 3
    assert all(("raises" in r or "lowers" in r) for r in res["reasons"])


def test_reasons_handle_missing_values(bundle):
    row = score.normalise_input(
        pd.DataFrame([{"country_name": "Uganda", "lat_deg": 1.0, "lon_deg": 32.0}])
    )
    items = explain.reasons(bundle["model"], row, bundle["background"], n_draws=20)
    text = [explain.sentence(i, row) for i in items]
    assert all("nan" not in t.lower() for t in text)


def test_triage_split_marks_requested_share(bundle, sample_rows):
    out = score.triage_split(score.score_frame(bundle, sample_rows), 0.1)
    assert out["visit_first"].sum() == 6


def test_permutation_importance_orders_groups(bundle, cleaned):
    imp = explain.permutation_importance(
        bundle["model"], cleaned.iloc[:1500], cleaned["no_water"].iloc[:1500], repeats=1
    )
    assert imp["auc_drop"].is_monotonic_decreasing
    assert set(imp["group"]) == set(explain.GROUPS)


def test_risk_map_renders(bundle, sample_rows):
    scored = score.score_frame(bundle, sample_rows)
    html = mapview.risk_map(scored).get_root().render()
    assert "leaflet" in html.lower()
    assert "<html" in mapview.risk_map(scored.iloc[0:0]).get_root().render().lower()


def test_app_functions_and_construction(bundle, sample_rows, tmp_path):
    md = app.check_point(
        bundle, "Uganda", 1.4, 32.3, "Borehole", "Hand Pump - Afridev", "Improved", 2005, "", ""
    )
    assert "Chance of no water" in md
    path = tmp_path / "in.csv"
    sample_rows.to_csv(path, index=False)
    table, out_csv, frame = app.rank_list(bundle, str(path), 0.2)
    assert len(table) == 60 and out_csv and "iframe" in frame
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    assert "missing columns" in app.rank_list(bundle, str(bad), 0.1)[0]["problem"].iloc[0]
    assert app.rank_list(bundle, None, 0.1)[1] is None
    assert app.build_app(bundle) is not None


def test_cli_score_writes_ranked_csv(trained, sample_rows, tmp_path):
    csv = tmp_path / "pts.csv"
    sample_rows.to_csv(csv, index=False)
    out = tmp_path / "ranked.csv"
    code = cli.main(
        ["score", str(csv), "--model", str(trained["out"] / "model.joblib"), "--out", str(out)]
    )
    assert code == 0
    ranked = pd.read_csv(out)
    assert len(ranked) == 60 and "visit_first" in ranked.columns
    assert np.isfinite(ranked["risk"]).all()


def test_load_bundle_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        score.load_bundle(tmp_path / "nope.joblib")


def test_metrics_file_is_strict_json(tmp_path):
    path = tmp_path / "m.json"
    study.write_metrics({"a": float("nan"), "b": [1.0, float("inf")], "c": np.float64(2.5)}, path)
    import json

    loaded = json.loads(path.read_text())
    assert loaded == {"a": None, "b": [1.0, None], "c": 2.5}
