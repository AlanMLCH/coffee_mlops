from datetime import date

import polars as pl
import pytest

from domains.coffee.analysis import (
    kind_agreement,
    kind_scores,
    market_history,
    market_summary,
    shop_kinds,
)
from mlops_core.analysis.studies import (
    categorical_profile,
    feature_recommendation,
    numeric_profile,
    residuals_by_group,
    target_distribution,
)
from mlops_core.config import DomainConfig, ModelSpec, TargetBands

SPEC = ModelSpec(
    target="total_cup_points",
    categorical=["country"],
    numeric=["altitude_m", "moisture_pct"],
    leakage=["aroma"],
)
BANDS = TargetBands(edges=[82, 85], labels=["low (<82)", "mid (82-85)", "high (>=85)"])


def features_frame() -> pl.DataFrame:
    """Two periods: the second scores higher, is Taiwan-heavy and sits higher up."""
    return pl.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(8)],
            "snapshot": ["old"] * 4 + ["new"] * 4,
            "country": [
                "Mexico",
                "Mexico",
                "Mexico",
                "Taiwan",
                "Taiwan",
                "Taiwan",
                "Mexico",
                "Taiwan",
            ],
            "altitude_m": [1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0],
            "moisture_pct": [11.0, None, 11.5, 12.0, 10.0, 10.5, None, 11.0],
            "total_cup_points": [80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0],
            "grading_date": [date(2018, 1, 1)] * 4 + [date(2023, 1, 1)] * 4,
        }
    )


def test_target_distribution_shows_the_shape_of_each_period() -> None:
    rows = target_distribution(
        features_frame(), "total_cup_points", "snapshot", "grading_date"
    ).rows(named=True)

    old, new = rows
    assert (old["snapshot"], old["n"], old["mean"]) == ("old", 4, 81.5)
    assert (new["snapshot"], new["n"], new["mean"]) == ("new", 4, 85.5)


def test_numeric_profile_reports_missingness_signal_and_drift() -> None:
    profile = numeric_profile(features_frame(), SPEC, "snapshot", "grading_date").rows(named=True)

    altitude = next(row for row in profile if row["feature"] == "altitude_m")
    moisture = next(row for row in profile if row["feature"] == "moisture_pct")
    assert altitude["correlation_with_target"] == pytest.approx(1.0)
    assert altitude["drift_sd"] > 1.0  # the periods are at different altitudes
    assert moisture["missing_pct"] == pytest.approx(25.0)


def test_categorical_profile_exposes_a_changing_mix() -> None:
    profile = categorical_profile(features_frame(), SPEC, "snapshot", "grading_date", min_rows=1)

    taiwan = profile.filter(pl.col("level") == "Taiwan").row(0, named=True)
    assert taiwan["share_first"] == pytest.approx(0.25)
    assert taiwan["share_last"] == pytest.approx(0.75)
    assert taiwan["share_change"] == pytest.approx(0.5)


def test_rare_levels_are_left_out_of_the_profile() -> None:
    frame = features_frame().with_columns(
        pl.when(pl.col("review_id") == "r0").then(pl.lit("Laos")).otherwise(pl.col("country"))
    )

    profile = categorical_profile(frame, SPEC, "snapshot", "grading_date", min_rows=2)

    assert "Laos" not in profile["level"].to_list()


def test_residuals_are_reported_by_group_and_by_quality_band() -> None:
    features = features_frame()
    # The predictions table carries the period, exactly as the batch job writes it.
    predictions = pl.DataFrame(
        {
            "review_id": features["review_id"],
            "snapshot": features["snapshot"],
            "prediction": [82.0] * 8,
        }
    )

    residuals = residuals_by_group(
        predictions,
        features,
        SPEC,
        "country",
        min_rows=1,
        period="snapshot",
        item_id="review_id",
        bands=BANDS,
    )

    assert set(residuals["kind"]) == {"group", "quality_band"}
    # Periods are never mixed: the model always looks good on the rows it trained on.
    high = residuals.filter((pl.col("level") == "high (>=85)") & (pl.col("snapshot") == "new")).row(
        0, named=True
    )
    assert high["n"] == 3  # 85, 86, 87, all in the newer period
    assert high["bias"] < 0  # the flat prediction under-rates the good lots
    assert residuals.filter(pl.col("snapshot") == "old")["n"].sum() > 0
    # A score on an edge belongs to the band above it: 82 is mid, not low.
    old = residuals.filter((pl.col("kind") == "quality_band") & (pl.col("snapshot") == "old"))
    assert dict(zip(old["level"], old["n"], strict=True)) == {"low (<82)": 2, "mid (82-85)": 2}


def test_band_edges_and_labels_must_agree() -> None:
    with pytest.raises(ValueError, match="2 edges make 3 bands"):
        TargetBands(edges=[82, 85], labels=["low", "high"])
    with pytest.raises(ValueError, match="ascending"):
        TargetBands(edges=[85, 82], labels=["a", "b", "c"])


@pytest.mark.parametrize(
    ("missing_pct", "signal", "drift_sd", "expected"),
    [
        pytest.param(80.0, 0.5, 0.1, "review: mostly missing", id="mostly-missing"),
        pytest.param(1.0, 0.5, 0.9, "review: distribution moved", id="drifted"),
        pytest.param(1.0, 0.01, 0.1, "review: little signal on its own", id="no-signal"),
        pytest.param(1.0, 0.5, 0.1, "keep", id="healthy"),
    ],
)
def test_the_suggested_action_explains_what_to_look_at(
    missing_pct: float, signal: float, drift_sd: float, expected: str
) -> None:
    numeric = pl.DataFrame(
        {
            "feature": ["altitude_m"],
            "missing_pct": [missing_pct],
            "correlation_with_target": [signal],
            "drift_sd": [drift_sd],
        }
    )
    empty_categorical = pl.DataFrame(
        schema={
            "feature": pl.String,
            "mean_target_first": pl.Float64,
            "share_change": pl.Float64,
        }
    )

    table = feature_recommendation(numeric, empty_categorical, importance=None)

    assert table["suggested_action"].to_list() == [expected]
    assert table["permutation_importance"].to_list() == [None]


def test_recommendations_carry_the_measured_importance() -> None:
    numeric = pl.DataFrame(
        {
            "feature": ["altitude_m"],
            "missing_pct": [1.0],
            "correlation_with_target": [0.5],
            "drift_sd": [0.1],
        }
    )
    empty = pl.DataFrame(
        schema={"feature": pl.String, "mean_target_first": pl.Float64, "share_change": pl.Float64}
    )
    importance = pl.DataFrame({"feature": ["altitude_m"], "permutation_importance": [0.12]})

    table = feature_recommendation(numeric, empty, importance)

    assert table["permutation_importance"].to_list() == [0.12]


def market_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country": ["Brazil", "Mexico", "Germany", "Mexico"],
            "market_year": [2025, 2025, 2025, 2024],
            "production": [60000.0, 4000.0, 0.0, 3900.0],
            "exports": [36000.0, 3400.0, 20000.0, 3300.0],
            "domestic_consumption": [22000.0, 3000.0, 9000.0, 2900.0],
            "imports": [100.0, 2400.0, 21000.0, 2300.0],
        }
    )


def test_market_summary_ranks_producers_and_ignores_pure_importers() -> None:
    summary = market_summary(market_frame(), year=2025, top=10)

    assert summary["country"].to_list() == ["Brazil", "Mexico"]  # Germany produces nothing
    assert summary["world_share_pct"].to_list() == pytest.approx([93.75, 6.25])
    mexico = summary.filter(pl.col("country") == "Mexico").row(0, named=True)
    assert mexico["imported_share_of_use"] == pytest.approx(0.8)


def test_market_history_follows_one_country_through_time() -> None:
    history = market_history(market_frame(), country="Mexico", since=2024)

    assert history["market_year"].to_list() == [2024, 2025]
    assert history["imported_share_of_use_pct"].to_list() == pytest.approx([79.31, 80.0], abs=0.01)


def test_studies_run_on_the_real_domain_config(coffee_config: DomainConfig) -> None:
    """The configured columns must exist in the frames the pipeline passes."""
    assert coffee_config.items.period == "snapshot"
    assert coffee_config.training.stratify_by in coffee_config.model.categorical


def shops_frame() -> pl.DataFrame:
    """Two registers: three DENUE places, two OSM ones, two pairs linked."""
    return pl.DataFrame(
        {
            "shop_id": ["denue-1", "denue-2", "denue-3", "osm-a", "osm-b"],
            "source": ["denue", "denue", "denue", "osm", "osm"],
            "kind": ["coffee", "unclassified", "juice", "coffee", "coffee"],
            "matched_shop_id": ["osm-a", "osm-b", None, "denue-1", "denue-2"],
        }
    )


def test_shop_kinds_share_each_register_by_kind() -> None:
    table = shop_kinds(shops_frame())

    osm = table.filter(pl.col("source") == "osm").row(0, named=True)
    assert (osm["kind"], osm["places"], osm["share_pct"]) == ("coffee", 2, 100.0)
    assert table.filter(pl.col("source") == "denue")["share_pct"].sum() == pytest.approx(100.0)


def test_the_name_rule_is_scored_on_the_places_both_registers_list() -> None:
    agreement = kind_agreement(shops_frame())
    scores = {row["metric"]: row for row in kind_scores(agreement).rows(named=True)}

    assert agreement["pairs"].sum() == 2  # the juice stand has no twin, so no verdict
    assert (scores["precision"]["hits"], scores["precision"]["of"]) == (1, 1)
    # OSM calls both coffee; the rule found one - the other's name said nothing.
    assert scores["recall"]["value"] == 0.5


def test_with_no_shared_places_there_is_no_score_rather_than_a_zero() -> None:
    lonely = shops_frame().with_columns(pl.lit(None, pl.String).alias("matched_shop_id"))

    scores = kind_scores(kind_agreement(lonely))

    assert scores["value"].to_list() == [None, None]
    assert scores["of"].to_list() == [0, 0]
