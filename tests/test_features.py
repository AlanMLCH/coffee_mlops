import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.features import MULTIPLE, add_coffee_origin, add_market_context, coffee_origins
from domains.coffee.schemas import SENSORY_COLUMNS
from mlops_core.data.clean import build_clean
from mlops_core.ml.features import build_features, features_schema, select_features
from mlops_core.storage import MANIFEST_NAME, read_table


def context_row(country: str, year: int, production: float, arabica: float) -> dict[str, object]:
    return {
        "country": country,
        "market_year": year,
        "production": production,
        "arabica_production": arabica,
        "exports": production / 2,
        "domestic_consumption": 100.0,
    }


CONTEXT = pl.DataFrame(
    [
        context_row("Mexico", 2021, 4000.0, 3600.0),
        context_row("Mexico", 2022, 4100.0, 3700.0),
        context_row("Taiwan", 2022, 0.0, 0.0),  # consumer, not producer
    ]
)


def test_items_see_the_previous_market_year_only() -> None:
    items = pl.DataFrame({"country": ["Mexico"], "grading_date": [date(2022, 9, 1)]})

    row = add_market_context(items, CONTEXT)

    # Graded in 2022 -> market year 2021, the latest one complete at grading time.
    assert row["ctx_production"].item() == 4000.0
    assert row["ctx_arabica_share"].item() == pytest.approx(0.9)


def test_shares_are_null_for_countries_without_production() -> None:
    items = pl.DataFrame({"country": ["Taiwan"], "grading_date": [date(2023, 3, 1)]})

    row = add_market_context(items, CONTEXT)

    assert row["ctx_production"].item() == 0.0
    assert row["ctx_arabica_share"].item() is None
    assert row["ctx_export_share"].item() is None


def test_items_without_context_keep_their_row() -> None:
    items = pl.DataFrame({"country": ["Myanmar"], "grading_date": [date(2023, 3, 1)]})

    assert add_market_context(items, CONTEXT)["ctx_production"].item() is None


@pytest.fixture
def clean_dir(coffee_adapter: CoffeeAdapter, raw_dir: Path) -> Path:
    build_clean(coffee_adapter, raw_dir.parent)
    return raw_dir.parent / "clean"


def test_feature_table_meets_its_contract_and_carries_no_leakage(
    coffee_adapter: CoffeeAdapter, clean_dir: Path
) -> None:
    review = coffee_adapter.config.model_named("review")
    enriched = coffee_adapter.enrich(
        "review",
        read_table(clean_dir / "coffee_reviews"),
        {"market_context": read_table(clean_dir / "market_context")},
    )
    features = select_features(enriched, review)

    features_schema(review).validate(features, lazy=True)
    assert not set(features.columns) & set(SENSORY_COLUMNS)
    assert features.height == 25


def test_build_features_writes_with_lineage_to_clean_partitions(
    coffee_adapter: CoffeeAdapter, clean_dir: Path
) -> None:
    path = build_features(
        coffee_adapter, "review", clean_dir.parent, at=datetime(2026, 9, 19, tzinfo=UTC)
    )

    manifest = json.loads((path.parent / MANIFEST_NAME).read_text())
    assert set(manifest["inputs"]) == {"coffee_reviews", "market_context"}
    assert all(p.startswith("built_at=") for p in manifest["inputs"].values())


def test_a_model_without_code_in_the_domain_is_named(coffee_adapter: CoffeeAdapter) -> None:
    """A model declared in the YAML but given no hooks is a config error, said plainly."""
    with pytest.raises(ValueError, match="no code for model 'tasting'"):
        coffee_adapter.context_tables("tasting")


def origin_row(coffee: str, **values: object) -> dict[str, object]:
    row: dict[str, object] = {
        "coffee_id": coffee,
        "country": None,
        "state": None,
        "processing_method": None,
        "producer": None,
        "varieties": None,
        "altitude_min_m": None,
        "altitude_max_m": None,
    }
    return row | values


ORIGINS = pl.DataFrame(
    [
        origin_row(
            "single",
            country="Mexico",
            state="Oaxaca",
            varieties=["typica"],
            processing_method="washed",
            altitude_min_m=1400.0,
            altitude_max_m=1500.0,
        ),
        # A blend: two arabicas from two states and a robusta, like Buna's Guarumbo.
        origin_row(
            "blend",
            country="Mexico",
            state="Oaxaca",
            varieties=["bourbon", "typica"],
            processing_method="washed",
            altitude_min_m=1400.0,
            altitude_max_m=1500.0,
        ),
        origin_row(
            "blend",
            country="Mexico",
            state="Chiapas",
            varieties=["bourbon"],
            processing_method="washed",
            altitude_min_m=700.0,
            altitude_max_m=700.0,
        ),
        origin_row("silent"),
    ],
    schema_overrides={"varieties": pl.List(pl.String)},
)


def test_a_coffee_is_what_its_origins_agree_on() -> None:
    summary = {row["coffee_id"]: row for row in coffee_origins(ORIGINS).iter_rows(named=True)}

    assert (summary["single"]["variety"], summary["single"]["altitude_m"]) == ("typica", 1450.0)
    # The blend agrees on country and process, not on state or variety.
    blend = summary["blend"]
    assert (blend["country"], blend["processing_method"]) == ("Mexico", "washed")
    assert (blend["state"], blend["variety"]) == (MULTIPLE, MULTIPLE)
    assert blend["altitude_m"] == (1450.0 + 700.0) / 2
    # A sheet that states nothing gives nothing: unknown, not "multiple".
    assert summary["silent"]["country"] is None and summary["silent"]["variety"] is None


def test_a_sheet_that_names_no_variety_says_nothing_about_varieties() -> None:
    """Zero would claim "this is not a Gesha"; the sheet never said that."""
    quiet = pl.DataFrame(
        [origin_row("quiet", country="Mexico")], schema_overrides={"varieties": pl.List(pl.String)}
    )

    summary = coffee_origins(quiet).row(0, named=True)

    assert summary["varieties_n"] is None
    assert summary["variety_gesha"] is None and summary["variety"] is None
    # A sheet that does list varieties says so for the ones it leaves out.
    listed = coffee_origins(ORIGINS.filter(pl.col("coffee_id") == "single")).row(0, named=True)
    assert (listed["variety_typica"], listed["variety_gesha"]) == (1.0, 0.0)


def test_offers_are_examples_only_with_a_price_to_learn_from() -> None:
    offers = pl.DataFrame(
        {
            "offer_id": ["a", "b", "c"],
            "coffee_id": ["single", "single", "single"],
            "price_mxn_per_kg": [1100.0, None, 8640.0],  # priced, a kit, a copied price
            "price_outlier": [False, None, True],
        }
    )

    enriched = add_coffee_origin(offers, ORIGINS)

    assert enriched["offer_id"].to_list() == ["a"]
    assert enriched.row(0, named=True)["state"] == "Oaxaca"


def test_a_request_keeps_the_origin_it_states() -> None:
    """Online, the caller describes the coffee: nothing in the catalogue overrides it,
    and the columns a sheet's summary would have produced are derived from what it says,
    or online and batch would feed the model different things."""
    request = pl.DataFrame(
        {"shop": ["buna"], "country": ["Kenya"], "variety": ["gesha"], "bag_grams": [340.0]}
    )

    enriched = add_coffee_origin(request, ORIGINS)

    assert enriched.select(request.columns).equals(request)  # stated, untouched
    row = enriched.row(0, named=True)
    assert (row["origins_n"], row["varieties_n"]) == (1.0, 1.0)
    assert (row["variety_gesha"], row["variety_typica"]) == (1.0, 0.0)


def test_a_request_that_names_no_variety_claims_nothing_about_them() -> None:
    request = pl.DataFrame({"shop": ["buna"], "variety": [None], "bag_grams": [340.0]})

    row = add_coffee_origin(request, ORIGINS).row(0, named=True)

    assert row["varieties_n"] is None and row["variety_gesha"] is None
