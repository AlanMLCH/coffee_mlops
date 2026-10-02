import math
from datetime import date

import polars as pl
import pytest

from domains.coffee.analysis import (
    borough_coffee_shops,
    coffee_and_schooling_figure,
    consumer_prices_by_borough,
    consumer_prices_by_fortnight,
    consumer_prices_by_state,
    flavor_clusters,
    flavor_prices,
    flavor_profiles,
    flavor_profiles_figure,
    green_coffee_figure,
    green_coffee_in_pesos,
    kind_agreement,
    kind_scores,
    market_history,
    market_summary,
    price_ladder,
    production_by_state,
    production_crosscheck,
    register_editions,
    roaster_coverage,
    shop_kinds,
    shop_turnover,
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


def test_a_model_without_categorical_features_has_an_empty_profile() -> None:
    numeric_only = SPEC.model_copy(update={"categorical": []})

    profile = categorical_profile(features_frame(), numeric_only, "snapshot", "grading_date", 1)
    numeric = numeric_profile(features_frame(), numeric_only, "snapshot", "grading_date")

    assert profile.is_empty() and "share_change" in profile.columns
    assert feature_recommendation(numeric, profile, None).height == len(numeric_only.numeric)


def test_rare_levels_are_left_out_of_the_profile() -> None:
    frame = features_frame().with_columns(
        pl.when(pl.col("review_id") == "r0").then(pl.lit("Laos")).otherwise(pl.col("country"))
    )

    profile = categorical_profile(frame, SPEC, "snapshot", "grading_date", min_rows=2)

    assert "Laos" not in profile["level"].to_list()


def test_a_single_period_has_nothing_to_drift_from() -> None:
    """One read of a catalogue: its studies must run, and say drift is unknown, not zero."""
    one_read = features_frame().filter(pl.col("snapshot") == "new")

    numeric = numeric_profile(one_read, SPEC, "snapshot", "grading_date")
    categorical = categorical_profile(one_read, SPEC, "snapshot", "grading_date", min_rows=2)

    assert numeric["drift_sd"].null_count() == numeric.height
    taiwan = categorical.filter(pl.col("level") == "Taiwan").row(0, named=True)
    assert (taiwan["n_first"], taiwan["share_change"]) == (3, 0.0)
    # Mexico has one row: counted once, it is below min_rows, not two rows at 2.
    assert "Mexico" not in categorical["level"].to_list()


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
    """The configured columns must exist in the frames the pipeline passes: a feature, or
    a key such as a group split's column."""
    for model in coffee_config.models:
        columns = {*model.keys, *model.spec.features}
        assert model.training.stratify_by in columns
        assert model.training.baseline_group in columns
    assert coffee_config.model_named("review").items.period == "snapshot"


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


def test_coffee_shops_are_counted_per_borough_against_area_and_residents() -> None:
    """The official register's coffee shops only; a borough without census figures keeps
    its count, without a rate per resident."""
    shops = pl.DataFrame(
        {
            "borough_id": ["b1", "b1", "b1", "b2", "b3"],
            "kind": ["coffee", "coffee", "juice", "coffee", "coffee"],
            "source": ["denue", "denue", "denue", "osm", "denue"],
        }
    )
    boroughs = pl.DataFrame(
        {
            "borough_id": ["b1", "b2", "b3"],
            "borough": ["Centro", "Norte", "Sur"],
            "area_km2": [2.0, 4.0, 10.0],
            "population": [20_000, 40_000, None],
            "schooling_years": [13.0, 11.0, None],
            "workplaces": [500, 100, None],
            "jobs_estimate": [4_000.0, 800.0, None],
        }
    )

    table = borough_coffee_shops(shops, boroughs)

    centro = table.row(0, named=True)
    assert (centro["borough"], centro["coffee_shops"], centro["per_km2"]) == ("Centro", 2, 1.0)
    assert centro["per_10k_people"] == 1.0
    # By day the centre holds 4,000 jobs for its 20,000 residents: two shops per 4,000.
    assert (centro["per_1k_jobs"], centro["jobs_per_resident"]) == (0.5, 0.2)
    assert table.filter(pl.col("borough") == "Norte")["coffee_shops"].item() == 0  # OSM's
    assert table.row(-1, named=True)["per_10k_people"] is None


def test_without_the_register_no_borough_has_a_count_rather_than_zero() -> None:
    """A clone without a DENUE token has only OpenStreetMap: zero shops would be a
    finding, and a figure of zeros would be published."""
    shops = pl.DataFrame({"borough_id": ["b1"], "kind": ["coffee"], "source": ["osm"]})
    boroughs = pl.DataFrame(
        {
            "borough_id": ["b1", "b2"],
            "borough": ["Centro", "Norte"],
            "area_km2": [2.0, 4.0],
            "population": [20_000, 40_000],
            "schooling_years": [13.0, 11.0],
            "workplaces": [None, None],
            "jobs_estimate": [None, None],
        },
        schema_overrides={"workplaces": pl.Int64, "jobs_estimate": pl.Float64},
    )

    table = borough_coffee_shops(shops, boroughs)

    assert table["coffee_shops"].null_count() == table["per_10k_people"].null_count() == 2
    assert table["per_1k_jobs"].null_count() == 2


def test_coffee_and_schooling_is_drawn_with_its_rank_correlation() -> None:
    table = pl.DataFrame(
        {
            "borough": ["A", "B", "C", "D"],
            "schooling_years": [10.0, 11.0, 12.0, None],
            "per_10k_people": [2.0, 3.0, 9.0, None],
        }
    )

    figure = coffee_and_schooling_figure(table)

    subtitle = figure.axes[0].texts[0].get_text()
    assert "3 boroughs" in subtitle and "Spearman 1.00" in subtitle
    assert {t.get_text() for t in figure.axes[0].texts[1:]} == {"A", "B", "C"}


def test_the_register_is_counted_by_the_edition_each_place_entered() -> None:
    shops = pl.DataFrame(
        {
            "source": ["denue", "denue", "denue", "osm"],
            "kind": ["coffee", "juice", "coffee", "coffee"],
            "listed_since": [date(2024, 11, 1), date(2024, 11, 1), date(2010, 7, 1), None],
        }
    )

    table = register_editions(shops)

    assert table.rows() == [
        (date(2010, 7, 1), 1, 1, pytest.approx(100 / 3)),
        (date(2024, 11, 1), 2, 1, pytest.approx(200 / 3)),
    ]


def test_turnover_counts_what_appeared_and_went_and_what_was_only_redrawn() -> None:
    """Between two reads of a map: a café deleted, a node redrawn as its building 20 m
    away under a new id, and a new place. The second register was read once: no pair."""

    def listed(snapshot: str, *places: tuple[str, str, float]) -> list[dict[str, object]]:
        return [
            {
                "source": "osm",
                "snapshot": snapshot,
                "shop_id": shop_id,
                "name": name,
                "latitude": 19.4,
                "longitude": longitude,
            }
            for shop_id, name, longitude in places
        ]

    history = pl.DataFrame(
        [
            *listed(
                "2026-09-22",
                ("n1", "Café Uno", -99.1),
                ("n2", "Café Dos", -99.2),
                ("n3", "Tres", -99.3),
            ),
            # "CAFE DOS" is the same name, folded; 0.0002 degrees of longitude is ~21 m.
            *listed(
                "2026-09-27",
                ("n1", "Café Uno", -99.1),
                ("w9", "CAFE DOS", -99.2002),
                ("n4", "Tres", -99.35),  # a namesake 5 km away is another place
            ),
            {
                "source": "denue",
                "snapshot": "2026-09-20",
                "shop_id": "d1",
                "name": None,
                "latitude": 19.4,
                "longitude": -99.1,
            },
        ]
    )

    table = shop_turnover(history, redraw_m=60.0)

    assert table.rows(named=True) == [
        {
            "source": "osm",
            "since": "2026-09-22",
            "until": "2026-09-27",
            "listed_before": 3,
            "listed_after": 3,
            "appeared": 2,
            "disappeared": 2,
            "redrawn": 1,
        }
    ]


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


def production_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "year": [2025, 2025, 2025],
            "state": ["Chiapas", "Chiapas", "Puebla"],
            "planted_ha": [100.0, 50.0, 30.0],
            "production_t": [300.0, 100.0, 100.0],
            "value_mxn": [1_500.0, 500.0, 1_400.0],
        }
    )


def test_production_is_shared_by_state_with_the_price_a_tonne_fetched() -> None:
    table = production_by_state(production_frame())

    chiapas = table.row(0, named=True)
    assert (chiapas["state"], chiapas["municipalities"], chiapas["share_pct"]) == (
        "Chiapas",
        2,
        80.0,
    )
    # Value over volume of the totals: 2,000 / 400, not the mean of the rows' prices.
    assert chiapas["rural_price_mxn_per_t"] == 5.0


def test_siap_cherry_is_set_against_psd_green_for_two_alignments() -> None:
    """The factor between them is shown, not assumed; so is the alignment question."""
    context = pl.DataFrame(
        {
            "country": ["Mexico", "Mexico", "Brazil"],
            "market_year": [2024, 2025, 2025],
            "production": [2.0, 4.0, 900.0],
        }
    )

    check = production_crosscheck(production_frame(), context, "Mexico")

    assert check["psd_market_year"].to_list() == [2024, 2025]
    assert check["psd_green_t"].to_list() == [120.0, 240.0]  # thousands of 60 kg bags
    assert check["cherry_per_green"].to_list() == pytest.approx([500 / 120, 500 / 240])


def test_roaster_coverage_counts_coffees_not_origins() -> None:
    """A blend gives a field once; a label no rule understood is not a method; the price
    per kilogram is counted over offers."""
    coffees = pl.DataFrame(
        {"shop": ["a", "a", "b"], "product_id": ["blend", "bare", "one"], "origins": [2, 0, 1]}
    ).with_columns(coffee_id=pl.concat_str("shop", "product_id", separator="-"))
    origins = pl.DataFrame(
        {
            "shop": ["a", "a", "b"],
            "product_id": ["blend", "blend", "one"],
            "country": ["Mexico", "Mexico", None],
            "processing_method": ["washed", "unclassified", "unclassified"],
        }
    ).with_columns(
        *[
            pl.lit(None, dtype=pl.String).alias(column)
            for column in ("state", "region", "producer", "farm", "process", "species")
        ],
        pl.lit(None, dtype=pl.Float64).alias("altitude_min_m"),
        pl.lit(None, dtype=pl.Float64).alias("sca_score"),
        pl.lit(None, dtype=pl.List(pl.String)).alias("varieties"),
    )
    offers = pl.DataFrame({"shop": ["a", "a", "b"], "price_mxn_per_kg": [900.0, None, 700.0]})
    # Two notes of one coffee: it gives tasting notes once.
    flavors = pl.DataFrame({"coffee_id": ["a-bare", "a-bare"], "note_en": ["peach", "honey"]})

    table = roaster_coverage(coffees, origins, offers, flavors)

    def given(shop: str, field: str) -> tuple[int, int]:
        row = table.filter((pl.col("shop") == shop) & (pl.col("field") == field))
        return row["given"].item(), row["of"].item()

    assert given("a", "sheet") == (1, 2)
    assert given("a", "country") == (1, 2)  # two origins, one coffee
    assert given("b", "processing method") == (0, 1)
    assert given("all", "processing method") == (1, 3)
    assert given("a", "price per kg") == (1, 2)
    assert given("all", "price per kg") == (2, 3)
    assert given("a", "tasting notes") == (1, 2)  # the one without a sheet still has notes
    assert given("all", "tasting notes") == (1, 3)
    assert table["shop"].unique(maintain_order=True).to_list() == ["a", "b", "all"]
    assert table.filter(pl.col("shop") == "a")["field"].to_list()[:2] == ["sheet", "country"]


def tasted() -> pl.DataFrame:
    """Twelve coffees: five Mexican washed ones taste of chocolate and caramel (one also
    of fruit), five natural ones from elsewhere of fruit and flowers (two also of spice),
    a blend of fruit and a coffee without a sheet of sugar. Shop b has four."""
    categories = {
        **{f"c{i}": ["nutty_cocoa", "sweet"] for i in range(1, 6)},
        **{f"c{i}": ["fruity", "floral"] for i in range(6, 11)},
        "c11": ["fruity"],
        "c12": ["sweet"],
    }
    categories["c1"] = [*categories["c1"], "fruity"]
    categories["c6"] = categories["c7"] = ["fruity", "floral", "spice"]
    return pl.DataFrame(
        [
            {"coffee_id": coffee, "shop": "a" if int(coffee[1:]) <= 8 else "b", "category": name}
            for coffee, names in categories.items()
            for name in names
        ]
    )


def sheets() -> pl.DataFrame:
    rows = [
        *[(f"c{i}", "Mexico", "washed") for i in range(1, 6)],
        *[(f"c{i}", "Ethiopia", "natural") for i in range(6, 11)],
        ("c11", "Brazil", "natural"),  # a blend: two origins, so neither counts
        ("c11", "Mexico", "washed"),
    ]
    return pl.DataFrame(rows, schema=["coffee_id", "country", "processing_method"], orient="row")


def test_flavor_profiles_share_each_group_by_category() -> None:
    table = flavor_profiles(tasted(), sheets(), "Mexico", min_rows=5)

    def share(dimension: str, group: str, category: str) -> float:
        row = table.filter(
            (pl.col("dimension") == dimension)
            & (pl.col("group") == group)
            & (pl.col("category") == category)
        )
        return float(row["share_pct"].item())

    groups = table.select("dimension", "group", "coffees").unique(maintain_order=True).rows()
    # The blend and the sheetless coffee count for everyone and their shop only; shop b
    # has four coffees, too few to speak for it.
    assert groups == [
        ("all", "every coffee", 12),
        ("origin", "Mexico", 5),
        ("origin", "elsewhere", 5),
        ("process", "natural", 5),
        ("process", "washed", 5),
        ("shop", "a", 8),
    ]
    assert share("all", "every coffee", "fruity") == pytest.approx(100 * 7 / 12)
    assert share("origin", "Mexico", "nutty_cocoa") == 100.0
    assert share("origin", "Mexico", "floral") == 0.0  # not named: zero, not missing
    assert share("process", "natural", "spice") == 40.0


def test_flavor_profiles_are_drawn_one_row_per_group() -> None:
    figure = flavor_profiles_figure(flavor_profiles(tasted(), sheets(), "Mexico", min_rows=5))

    rows = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    columns = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert rows[:2] == ["every coffee (12)", "from Mexico (5)"]
    assert "natural process (5)" in rows
    assert columns[0] == "fruity" and "nutty/\ncocoa" in columns


def priced(
    coffee: str, per_kg: float, grams: float = 250.0, outlier: bool = False
) -> dict[str, object]:
    return {
        "coffee_id": coffee,
        "shop": "a" if int(coffee[1:]) <= 8 else "b",
        "bag_grams": grams,
        "price_mxn_per_kg": per_kg,
        "price_outlier": outlier,
    }


def test_flavor_prices_compare_within_a_shop_and_a_size_with_a_family_wise_interval() -> None:
    offers = pl.DataFrame(
        [
            *[priced(f"c{i}", 1000.0) for i in range(1, 6)],
            *[priced(f"c{i}", 1200.0) for i in range(6, 9)],
            priced("c9", 1200.0),
            priced("c10", 1200.0),
            priced("c11", 1000.0),
            priced("c12", 1000.0),
            priced("c1", 5000.0, outlier=True),  # a price copied from another size: out
            priced("c1", 800.0, grams=1000.0),  # alone in its size: at its own median
        ]
    )

    table = flavor_prices(tasted(), offers, min_rows=5)

    # Spice has two coffees: too few to compare. Four comparisons, so each interval is
    # at 1 - 0.05/4.
    assert set(table["category"]) == {"floral", "fruity", "nutty_cocoa", "sweet"}
    assert table["family_level"].unique().to_list() == [pytest.approx(0.9875)]
    floral = table.filter(pl.col("category") == "floral").row(0, named=True)
    # Shop a's median is 1,000 and shop b's 1,100: the floral coffees sit at 1.2 and
    # 12/11 of theirs, the others at 1 (c1's two sizes) and 10/11.
    inside = (3 * math.log(1.2) + 2 * math.log(12 / 11)) / 5
    outside = 2 * math.log(10 / 11) / 7
    assert (floral["coffees"], floral["others"]) == (5, 7)
    assert floral["premium_pct"] == pytest.approx(100 * (math.exp(inside - outside) - 1))
    assert floral["ci_low_pct"] < floral["premium_pct"] < floral["ci_high_pct"]
    assert floral["probability_dearer"] == 1.0
    assert table["premium_pct"].to_list() == sorted(table["premium_pct"], reverse=True)


def test_flavor_clusters_find_two_clear_types_and_say_how_clear() -> None:
    two_types = pl.DataFrame(
        [
            {"coffee_id": f"{kind}{i}", "category": category}
            for kind, categories in (("x", ["fruity", "floral"]), ("y", ["nutty_cocoa", "sweet"]))
            for i in range(5)
            for category in categories
        ]
    )

    table = flavor_clusters(two_types)

    # Two distinct sets of categories: no more than two clusters can be formed.
    assert table.select("k", "cluster", "coffees", "profile").rows() == [
        (2, 1, 5, "floral 100%, fruity 100%"),  # as large as the other: by its profile
        (2, 2, 5, "nutty_cocoa 100%, sweet 100%"),
    ]
    assert table["silhouette"].to_list() == [1.0, 1.0]
    assert table["structure"].to_list() == ["strong", "strong"]
    assert table["chosen"].all()


def test_flavor_clusters_read_a_weak_structure_as_weak() -> None:
    table = flavor_clusters(tasted())

    assert table["k"].unique().to_list() == list(range(2, 7))
    assert table.filter(pl.col("chosen"))["k"].n_unique() == 1
    assert set(table["structure"]) <= {"none", "weak", "reasonable", "strong"}
    for k, clusters in table.group_by("k"):
        assert clusters["coffees"].sum() == 12 and clusters["cluster"].to_list() == list(
            range(1, k[0] + 1)
        )


def test_no_notes_no_flavor_studies() -> None:
    empty = tasted().clear()

    assert flavor_profiles(empty, sheets(), "Mexico", min_rows=5).is_empty()
    assert flavor_prices(empty, pl.DataFrame([priced("c1", 1.0)]), min_rows=5).is_empty()
    assert flavor_clusters(empty).is_empty()


CITY = "Ciudad de México"


def shelf_prices() -> pl.DataFrame:
    """Five prices: three in the city (two stores, one borough each), two in Chiapas."""
    rows = [
        # store, state, borough, product, sweetened, decaf, per kg, day
        ("Walmart Polanco", CITY, "Miguel Hidalgo", "ground", False, False, 400.0, 3),
        ("Walmart Polanco", CITY, "Miguel Hidalgo", "ground", True, False, 250.0, 3),
        ("Soriana Coyoacán", CITY, "Coyoacán", "instant", False, False, 900.0, 20),
        ("Chedraui Tapachula", "Chiapas", None, "ground", False, False, 380.0, 3),
        ("Chedraui Tapachula", "Chiapas", None, "instant", False, True, 950.0, 3),
    ]
    return pl.DataFrame(
        [
            {
                "store": store, "state": state, "borough": borough,
                "borough_id": None if borough is None else f"id-{borough}",
                "product": product, "sweetened": sweetened, "decaf": decaf,
                "price_mxn_per_kg": per_kg, "date": date(2026, 7, day),
                "fortnight": date(2026, 7, 1 if day <= 15 else 16),
                "latitude": 19.0, "longitude": -99.0,
            }
            for store, state, borough, product, sweetened, decaf, per_kg, day in rows
        ]
    )  # fmt: skip


def cherry() -> pl.DataFrame:
    """Two years of SIAP: only the latest one prices the cherry."""
    return pl.DataFrame(
        {
            "year": [2024, 2025, 2025],
            "state": ["Chiapas", "Chiapas", "Puebla"],
            "production_t": [10.0, 100.0, 100.0],
            "value_mxn": [1.0, 500_000.0, 1_500_000.0],
        }
    )


def roaster_bags() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "price_mxn_per_kg": [1000.0, 1200.0, 5000.0, None],
            "price_outlier": [False, None, True, None],  # a copied price is left out
            "observed_on": [date(2026, 9, 27)] * 4,
        }
    )


def test_shelf_prices_are_followed_by_fortnight_in_the_country_and_the_city() -> None:
    table = consumer_prices_by_fortnight(shelf_prices(), CITY)

    ground = table.filter(pl.col("line") == "ground")
    assert ground.select("scope", "median_mxn_per_kg", "prices", "stores").rows() == [
        ("city", 400.0, 1, 1),
        ("national", 390.0, 2, 2),
    ]
    assert set(table["line"]) == {"ground", "ground, sweetened", "instant", "instant, decaf"}
    assert table.filter(pl.col("line") == "instant")["fortnight"].to_list() == [
        date(2026, 7, 16), date(2026, 7, 16),
    ]  # fmt: skip


def test_a_borough_is_summed_up_from_the_city_s_prices_only() -> None:
    table = consumer_prices_by_borough(shelf_prices())

    assert table.select("borough", "line", "median_mxn_per_kg").rows() == [
        ("Coyoacán", "instant", 900.0),
        ("Miguel Hidalgo", "ground", 400.0),
        ("Miguel Hidalgo", "ground, sweetened", 250.0),
    ]


def test_a_state_s_shelf_is_set_beside_what_its_growers_were_paid() -> None:
    states = {"chiapas": "Chiapas", "estado de mexico": "México"}

    table = consumer_prices_by_state(shelf_prices(), cherry(), states)

    # Plain coffee only: the sweetened and the decaf are other products.
    assert table.select("state", "product", "median_mxn_per_kg").rows() == [
        ("Ciudad de México", "ground", 400.0),  # dearest first, product by product
        ("Chiapas", "ground", 380.0),
        ("Ciudad de México", "instant", 900.0),
    ]
    chiapas = table.filter(pl.col("state") == "Chiapas").row(0, named=True)
    assert (chiapas["cherry_mxn_per_kg"], chiapas["siap_year"]) == (5.0, 2025)  # 2025's only
    assert table.filter(pl.col("state") == CITY)["cherry_mxn_per_kg"].is_null().all()


def green_prices() -> pl.DataFrame:
    """Two months of both indicators, a daily row that is not a month, and a month from
    before the peso-dollar series."""
    return pl.DataFrame(
        {
            "period": [date(2026, 7, 1), date(2026, 8, 1), date(2026, 8, 1),
                       date(2026, 8, 3), date(1990, 1, 1)],
            "frequency": ["monthly", "monthly", "monthly", "daily", "monthly"],
            "indicator": ["other_milds", "other_milds", "robustas", "other_milds", "robustas"],
            "usd_cents_per_lb": [100 * 0.45359237 * 8.0, 100 * 0.45359237 * 7.0,
                                 100 * 0.45359237 * 4.0, 999.0, 50.0],
        }
    )  # fmt: skip


def rates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2026, 7, 1), date(2026, 7, 2), date(2026, 8, 3)],
            "mxn_per_usd": [17.0, 18.0, 20.0],
        }
    )


def test_green_coffee_is_put_in_pesos_at_the_month_s_mean_rate() -> None:
    green = green_coffee_in_pesos(green_prices(), rates())

    # $8/kg at July's mean of 17.5; the daily row and the month without rates are left out.
    assert green.select("period", "indicator", "rate_days", "mxn_per_kg").rows() == [
        (date(2026, 7, 1), "other_milds", 2, pytest.approx(140.0)),
        (date(2026, 8, 1), "other_milds", 1, pytest.approx(140.0)),
        (date(2026, 8, 1), "robustas", 1, pytest.approx(80.0)),
    ]


def test_green_coffee_is_drawn_with_whichever_indicators_it_has() -> None:
    """A workbook that stopped publishing one series still draws the other."""
    milds = green_coffee_in_pesos(green_prices(), rates()).filter(
        pl.col("indicator") == "other_milds"
    )

    figure = green_coffee_figure(milds)

    labels = {text.get_text().strip() for text in figure.axes[0].texts}
    assert "other mild Arabicas" in labels and "Robustas" not in labels
    assert len(figure.axes[0].lines) == 1


def test_the_ladder_prices_a_kilogram_at_each_step_in_its_own_unit() -> None:
    green = green_coffee_in_pesos(green_prices(), rates())

    ladder = price_ladder(shelf_prices(), roaster_bags(), cherry(), green, CITY)

    assert ladder.select("step", "unit", "measure", "mxn_per_kg", "observations").rows() == [
        ("cherry at the farm gate", "kg of coffee cherry", "value over volume", 10.0, 2),
        ("green coffee at the port", "kg of green coffee", "the month's price",
         pytest.approx(140.0), 1),
        ("supermarket, ground + sugar", "kg of ground coffee and sugar", "median", 250.0, 1),
        ("supermarket, ground", "kg of ground coffee", "median", 400.0, 1),
        ("supermarket, instant", "kg of instant coffee", "median", 900.0, 1),
        ("specialty roaster", "kg of roasted coffee", "median", 1100.0, 2),
    ]  # fmt: skip
    # The latest month of the group Mexico's Arabica is priced in.
    assert ladder.filter(pl.col("source") == "World Bank, FRED")["period"].item() == "2026-08"
    assert ladder.filter(pl.col("source") == "PROFECO")["period"].to_list()[0] == (
        "2026-07-03 to 2026-07-03"
    )


def test_a_ladder_with_nothing_to_stand_on_is_empty_not_an_error() -> None:
    nothing = shelf_prices().clear()

    no_green = green_coffee_in_pesos(green_prices(), rates().clear())

    ladder = price_ladder(nothing, roaster_bags().clear(), cherry().clear(), no_green, CITY)

    assert ladder.is_empty()
    assert "mxn_per_kg" in ladder.columns
