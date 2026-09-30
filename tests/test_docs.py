"""Documentation drifts the moment nobody checks it.

These tests do not judge prose; they check that the data dictionary still describes the
contracts the code enforces, so a renamed or added column cannot ship undocumented.
"""

import re
from pathlib import Path

import pytest

from domains.coffee.schemas import (
    BOROUGHS,
    CONSUMER_PRICES,
    EXCHANGE_RATES,
    MARKET_CONTEXT,
    MEXICO_PRODUCTION,
    PRICE_INDICATORS,
    ROASTER_COFFEES,
    ROASTER_OFFER_HISTORY,
    ROASTER_OFFERS,
    coffee_reviews_schema,
    coffee_shops_schema,
    roaster_flavors_schema,
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
# The detailed record, stage by stage, with the full architecture diagram: private, kept in
# a working copy and out of the repository. The README is the public summary, with a
# diagram of the main pieces only. The plan's checks run wherever it exists.
DEVELOPMENT_PLAN = DOCS / "development_plan.md"
# What anyone who clones the repository reads.
PUBLIC_DOCUMENTS = [
    "README.md",
    *(f"docs/{path.name}" for path in sorted(DOCS.glob("*.md")) if path != DEVELOPMENT_PLAN),
]


def private_plan() -> str:
    if not DEVELOPMENT_PLAN.exists():
        pytest.skip("the development plan is private: it lives in a working copy, not the repo")
    return DEVELOPMENT_PLAN.read_text(encoding="utf-8")


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
        "price_indicators": list(PRICE_INDICATORS.columns),
        "consumer_prices": list(CONSUMER_PRICES.columns),
        "exchange_rates": list(EXCHANGE_RATES.columns),
        "roaster_offer_history": list(ROASTER_OFFER_HISTORY.columns),
        "roaster_flavors": list(roaster_flavors_schema(coffee_config.cleaning).columns),
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
        "price_indicators",
        "consumer_prices",
        "exchange_rates",
        "roaster_offer_history",
        "roaster_origin_history",
        "roaster_flavors",
        "review_features",
        "offer_features",
        "green_price_features",
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
    return "\n".join(re.findall(r"```mermaid\n(.*?)```", private_plan(), flags=re.DOTALL))


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

    assert diagram, "the development plan has no architecture diagram"
    assert missing == []


@pytest.mark.parametrize("document", ["README.md", "docs/sources.md", "docs/development_plan.md"])
def test_every_relative_link_points_at_a_file(document: str) -> None:
    """A figure or a page moved without its links is a broken page on GitHub, and nothing
    else would notice."""
    path = DOCS.parent / document
    text = private_plan() if path == DEVELOPMENT_PLAN else path.read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text)
    local = [link for link in links if not link.startswith(("http://", "https://"))]

    assert local  # the documents link to each other and to their figures
    assert [link for link in local if not (path.parent / link).exists()] == []


@pytest.mark.parametrize("document", PUBLIC_DOCUMENTS)
def test_no_public_document_points_at_the_private_plan(document: str) -> None:
    """The plan stays out of the repository: a link to it works in the working copy that
    has it and is broken for everyone who clones - which the link check above, run here,
    cannot see."""
    assert "development_plan.md" not in (DOCS.parent / document).read_text(encoding="utf-8")
