"""Documentation drifts the moment nobody checks it.

These tests do not judge prose; they check that the data dictionary still describes the
contracts the code enforces, so a renamed or added column cannot ship undocumented.
"""

import re
from pathlib import Path

import pytest

from domains.coffee.schemas import (
    BOROUGHS,
    MARKET_CONTEXT,
    MEXICO_PRODUCTION,
    ROASTER_COFFEES,
    ROASTER_OFFERS,
    coffee_reviews_schema,
    coffee_shops_schema,
    roaster_origins_schema,
)
from mlops_core.adapter import available_domains, domain_dir, load_adapter
from mlops_core.config import DomainConfig
from mlops_core.ml.features import features_schema
from mlops_core.ml.predict import predictions_schema

DOCS = Path(__file__).resolve().parents[1] / "docs"
# Beside the domain's code: the agent reads it as the schema its SQL is written against.
DATA_DICTIONARY = (domain_dir("coffee") / "data_dictionary.md").read_text(encoding="utf-8")
MODEL_CARD = (DOCS / "model-card.md").read_text(encoding="utf-8")
README = (DOCS.parent / "README.md").read_text(encoding="utf-8")


def schema_columns(coffee_config: DomainConfig) -> dict[str, list[str]]:
    return {
        "coffee_reviews": list(coffee_reviews_schema(coffee_config.cleaning).columns),
        "market_context": list(MARKET_CONTEXT.columns),
        "boroughs": list(BOROUGHS.columns),
        "mexico_production": list(MEXICO_PRODUCTION.columns),
        "coffee_shops": list(coffee_shops_schema(coffee_config.cleaning).columns),
        "roaster_coffees": list(ROASTER_COFFEES.columns),
        "roaster_origins": list(roaster_origins_schema(coffee_config.cleaning).columns),
        "roaster_offers": list(ROASTER_OFFERS.columns),
        **{
            model.features_table: list(features_schema(model).columns)
            for model in coffee_config.models
        },
        **{
            model.predictions_table: list(predictions_schema(model).columns)
            for model in coffee_config.models
        },
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
    "table",
    [
        "coffee_reviews",
        "market_context",
        "boroughs",
        "coffee_shops",
        "mexico_production",
        "roaster_coffees",
        "roaster_origins",
        "roaster_offers",
        "review_features",
        "offer_features",
    ],
)
def test_each_table_has_its_own_section(table: str) -> None:
    assert f"`{table}`" in DATA_DICTIONARY or f".{table}`" in DATA_DICTIONARY


def test_the_model_card_names_the_features_the_model_actually_uses(
    coffee_config: DomainConfig,
) -> None:
    spec = coffee_config.model_named("review").spec
    undocumented = [f for f in spec.features if f"`{f}`" not in MODEL_CARD]

    assert undocumented == []


def test_the_model_card_states_the_leakage_rule(coffee_config: DomainConfig) -> None:
    # The one thing a reader must not have to discover on their own.
    assert "excluded by contract" in MODEL_CARD
    leakage = coffee_config.model_named("review").spec.leakage
    assert all(score in MODEL_CARD for score in leakage[:3])


def architecture_diagram() -> str:
    return "\n".join(re.findall(r"```mermaid\n(.*?)```", README, flags=re.DOTALL))


@pytest.mark.parametrize("domain", available_domains())
def test_the_architecture_diagram_shows_every_source_and_table(domain: str) -> None:
    """The diagram is how a reader learns the flow, and one that quietly forgot a source
    is wrong in a way nobody notices: it has to change in the feature that adds one."""
    diagram = architecture_diagram()
    adapter = load_adapter(domain)
    names = {
        *adapter.config.sources,
        *adapter.json_readers(),
        *adapter.clean_contracts(),
        *adapter.config.corpus_tables,
        *[model.features_table for model in adapter.config.models],
        *[model.predictions_table for model in adapter.config.models],
    }

    # Declared as a node - the name followed by its shape - not merely mentioned: a style
    # line such as `class ...,boroughs,... domain` would otherwise keep a deleted node
    # looking present.
    declared = {m.group(1) for m in re.finditer(r"(?<![\w])(\w+)(?=\[|\(|\{)", diagram)}
    missing = sorted(names - declared)

    assert diagram, "the README has no architecture diagram"
    assert missing == []
