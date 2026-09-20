import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from coffee_mlops.config import DomainConfig
from coffee_mlops.data.clean import build_clean
from coffee_mlops.data.schemas import SENSORY_COLUMNS
from coffee_mlops.ml.features import (
    add_market_context,
    build_features,
    build_review_features,
    review_features_schema,
)
from coffee_mlops.storage import MANIFEST_NAME, read_table


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
def clean_dir(coffee_config: DomainConfig, raw_dir: Path) -> Path:
    build_clean(coffee_config, raw_dir.parent)
    return raw_dir.parent / "clean"


def test_feature_table_meets_its_contract_and_carries_no_leakage(
    coffee_config: DomainConfig, clean_dir: Path
) -> None:
    features = build_review_features(
        read_table(clean_dir / "coffee_reviews"),
        read_table(clean_dir / "market_context"),
        coffee_config.model,
    )

    review_features_schema(coffee_config.model).validate(features, lazy=True)
    assert not set(features.columns) & set(SENSORY_COLUMNS)
    assert features.height == 25


def test_build_features_writes_with_lineage_to_clean_partitions(
    coffee_config: DomainConfig, clean_dir: Path
) -> None:
    path = build_features(coffee_config, clean_dir.parent, at=datetime(2026, 9, 19, tzinfo=UTC))

    manifest = json.loads((path.parent / MANIFEST_NAME).read_text())
    assert set(manifest["inputs"]) == {"coffee_reviews", "market_context"}
    assert all(p.startswith("built_at=") for p in manifest["inputs"].values())
