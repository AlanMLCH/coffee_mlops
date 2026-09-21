"""Documentation drifts the moment nobody checks it.

These tests do not judge prose; they check that the data dictionary still describes the
contracts the code enforces, so a renamed or added column cannot ship undocumented.
"""

from pathlib import Path

import pytest

from mlops_core.config import DomainConfig
from mlops_core.data.schemas import (
    BOROUGHS,
    COFFEE_SHOPS,
    MARKET_CONTEXT,
    coffee_reviews_schema,
)
from mlops_core.ml.features import review_features_schema
from mlops_core.ml.predict import PREDICTIONS

DOCS = Path(__file__).resolve().parents[1] / "docs"
DATA_DICTIONARY = (DOCS / "data-dictionary.md").read_text(encoding="utf-8")
MODEL_CARD = (DOCS / "model-card.md").read_text(encoding="utf-8")


def schema_columns(coffee_config: DomainConfig) -> dict[str, list[str]]:
    return {
        "coffee_reviews": list(coffee_reviews_schema(coffee_config.cleaning).columns),
        "market_context": list(MARKET_CONTEXT.columns),
        "boroughs": list(BOROUGHS.columns),
        "coffee_shops": list(COFFEE_SHOPS.columns),
        "review_features": list(review_features_schema(coffee_config.model).columns),
        "review_predictions": list(PREDICTIONS.columns),
    }


def test_every_column_of_every_table_is_documented(coffee_config: DomainConfig) -> None:
    missing = {
        f"{table}.{column}"
        for table, columns in schema_columns(coffee_config).items()
        for column in columns
        if f"`{column}`" not in DATA_DICTIONARY
    }

    assert missing == set()


@pytest.mark.parametrize(
    "table", ["coffee_reviews", "market_context", "boroughs", "coffee_shops", "review_features"]
)
def test_each_table_has_its_own_section(table: str) -> None:
    assert f"`{table}`" in DATA_DICTIONARY or f".{table}`" in DATA_DICTIONARY


def test_the_model_card_names_the_features_the_model_actually_uses(
    coffee_config: DomainConfig,
) -> None:
    undocumented = [f for f in coffee_config.model.features if f"`{f}`" not in MODEL_CARD]

    assert undocumented == []


def test_the_model_card_states_the_leakage_rule(coffee_config: DomainConfig) -> None:
    # The one thing a reader must not have to discover on their own.
    assert "excluded by contract" in MODEL_CARD
    assert all(score in MODEL_CARD for score in coffee_config.model.leakage[:3])
