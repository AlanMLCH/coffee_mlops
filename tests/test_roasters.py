"""The roasters' shops, from recordings of what each one actually served.

Four shops, two platforms, three ways of saying what is coffee, and one shop whose
attributes live only on its product pages - the tests hold each of those.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

import domains.coffee
from domains.coffee.config import RoastersConfig
from domains.coffee.roaster_sheets import clean_roasters
from domains.coffee.schemas import ROASTER_CATALOGS, clean_schemas
from domains.coffee.sources.roasters import ingest_catalogs, to_frame
from mlops_core.contracts import check_contract
from mlops_core.data.api import ApiClient
from mlops_core.data.robots import RobotsPolicy
from tests.fakes import shop_response, without_rate_limits

AGENT = "coffee-mlops/0.1.0 (+https://github.com/AlanMLCH/coffee_mlops)"
NOW = datetime(2026, 9, 21, tzinfo=UTC)


class Shops:
    """The recorded shops, optionally with one robots.txt replaced."""

    def __init__(self, robots: dict[str, str] | None = None, reverse: bool = False) -> None:
        self.robots = robots or {}
        self.reverse = reverse  # list every catalog backwards, as a shop is free to
        self.asked: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(f"{request.url.host}{request.url.path}")
        if request.url.path == "/robots.txt" and request.url.host in self.robots:
            return httpx.Response(200, text=self.robots[request.url.host])
        response = shop_response(request)
        assert response is not None, request.url
        if self.reverse and response.headers.get("content-type") == "application/json":
            payload = response.json()
            for listing in ("products", "items"):
                if listing in payload:
                    payload[listing].reverse()
            return httpx.Response(200, json=payload)
        return response


@pytest.fixture
def roasters(coffee_config) -> RoastersConfig:  # type: ignore[no-untyped-def]
    config = without_rate_limits(coffee_config).roasters
    assert config is not None
    return config


def ingest(
    shops: Shops, config: RoastersConfig, tmp_path: Path, cache: str = "cache"
) -> tuple[dict, dict[str, str]]:  # type: ignore[type-arg]
    """Read every shop into `tmp_path/raw`, through a cache of its own."""
    http = httpx.Client(transport=httpx.MockTransport(shops.handler))
    clients = {
        shop.shop: ApiClient(
            client=http, cache_dir=tmp_path / cache / shop.shop, min_interval_s=0.0
        )
        for shop in config.shops
    }
    policies = {name: RobotsPolicy(client, AGENT) for name, client in clients.items()}
    artifact, skipped = ingest_catalogs(clients, policies, config, tmp_path / "raw", NOW)
    return json.loads(artifact.path.read_text(encoding="utf-8")), skipped


def test_each_shop_keeps_only_its_coffee(roasters: RoastersConfig, tmp_path: Path) -> None:
    """By type (Almanegra), by tag (Buna), everything (Jiribilla), all but merch (Cucurucho)."""
    document, skipped = ingest(Shops(), roasters, tmp_path)

    kept = {shop: [p["title"] for p in c["products"]] for shop, c in document["shops"].items()}
    assert skipped == {}
    assert {shop: len(titles) for shop, titles in kept.items()} == {
        "almanegra": 3,  # of 4: the mug is not coffee
        "buna": 2,  # of 3: only the listings tagged "Café"
        "jiribilla": 3,  # sells only coffee
        "cucurucho": 3,  # of 5: the tote bag and the subscription are out
    }
    assert not any("Tote" in title or "Suscripci" in title for title in kept["cucurucho"])


def test_product_pages_are_read_only_where_the_attributes_live(
    roasters: RoastersConfig, tmp_path: Path
) -> None:
    shops = Shops()
    document, _ = ingest(shops, roasters, tmp_path)

    buna_pages = document["shops"]["buna"]["pages"]
    assert len(buna_pages) == 2 and all(page.startswith("<html>") for page in buna_pages.values())
    assert not document["shops"]["almanegra"]["pages"]
    assert not any(asked.startswith("almanegra.cafe/products/") for asked in shops.asked)


def test_a_shop_whose_robots_says_no_is_skipped_out_loud(
    roasters: RoastersConfig, tmp_path: Path
) -> None:
    """Never read around a refusal: the others are still stored, and the skip recorded."""
    shops = Shops(robots={"buna.mx": "User-agent: *\nDisallow: /\n"})

    document, skipped = ingest(shops, roasters, tmp_path)

    assert list(skipped) == ["buna"] and "robots.txt disallows" in skipped["buna"]
    assert "buna" not in document["shops"] and document["skipped"] == skipped
    assert shops.asked.count("buna.mx/robots.txt") == 1
    assert not any(asked.startswith("buna.mx/products") for asked in shops.asked)


def test_offers_meet_the_raw_contract_across_both_platforms(
    roasters: RoastersConfig, tmp_path: Path
) -> None:
    document, _ = ingest(Shops(), roasters, tmp_path)

    offers = check_contract(ROASTER_CATALOGS, to_frame(document))

    by_platform = offers.group_by("platform").agg(
        pl.col("platform_grams").null_count().alias("no_grams"), pl.len()
    )
    squarespace = by_platform.filter(pl.col("platform") == "squarespace").row(0, named=True)
    assert squarespace["no_grams"] == squarespace["len"]  # its weight unit is not trusted
    assert offers.filter(pl.col("shop") == "cucurucho")["price"].min() > 0  # priceMoney read
    assert offers["url"].str.starts_with("https://").all()


def test_the_same_catalog_in_another_order_is_not_new_data(
    roasters: RoastersConfig, tmp_path: Path
) -> None:
    """Shops list products in whatever order; stored in a canonical one, nothing changes."""
    ingest(Shops(), roasters, tmp_path, cache="first")
    ingest(Shops(reverse=True), roasters, tmp_path, cache="second")  # asked afresh, backwards

    assert len(list((tmp_path / "raw" / "roaster_catalogs").iterdir())) == 1


@pytest.fixture
def clean_tables(roasters: RoastersConfig, tmp_path: Path) -> dict[str, pl.DataFrame]:
    """The recorded shops, read and cleaned, each table held to its contract."""
    document, _ = ingest(Shops(), roasters, tmp_path)
    rules = domains.coffee.adapter().config.cleaning
    offers = check_contract(ROASTER_CATALOGS, to_frame(document))
    tables = clean_roasters(offers, rules, NOW)
    contracts = clean_schemas(rules)
    return {name: check_contract(contracts[name], table) for name, table in tables.items()}


def origins_of(tables: dict[str, pl.DataFrame], title: str) -> list[dict]:  # type: ignore[type-arg]
    coffee = tables["roaster_coffees"].filter(pl.col("title") == title)
    return (
        tables["roaster_origins"]
        .join(coffee.select("shop", "product_id"), on=["shop", "product_id"])
        .sort("origin")
        .to_dicts()
    )


def test_a_blend_gets_one_row_per_origin(clean_tables: dict[str, pl.DataFrame]) -> None:
    """Buna's Guarumbo page lists three components with the same headings."""
    origins = origins_of(clean_tables, "Café Guarumbo")

    assert [(o["state"], o["species"]) for o in origins] == [
        ("Oaxaca", "arabica"),
        ("Chiapas", "arabica"),
        ("Chiapas", "robusta"),
    ]
    assert origins[2]["altitude_min_m"] == 700
    assert origins_of(clean_tables, "Micha y Micha") == []  # its page has no sheet


def test_a_description_sheet_is_read_into_canonical_values(
    clean_tables: dict[str, pl.DataFrame],
) -> None:
    (yemen,) = origins_of(clean_tables, "Yemen Mokha Haimi")

    assert (yemen["country"], yemen["altitude_min_m"], yemen["altitude_max_m"]) == (
        "Yemen",
        2000,
        2400,
    )
    assert yemen["varieties"] == ["heirloom"] and yemen["processing_method"] == "natural"


def test_squarespace_descriptions_are_read_from_the_excerpt(
    clean_tables: dict[str, pl.DataFrame],
) -> None:
    """Cucurucho's `body` is empty; its text, labels included, is in `excerpt`."""
    (chiapas,) = origins_of(clean_tables, "Chiapas- Caramelo, avellana y chocolate")
    coffees = clean_tables["roaster_coffees"].filter(pl.col("shop") == "cucurucho")

    assert (chiapas["country"], chiapas["state"], chiapas["altitude_min_m"]) == (
        "Mexico",
        "Chiapas",
        1250,
    )
    assert coffees["description"].is_not_null().all()


def test_the_size_in_the_title_beats_the_platforms_weight(
    clean_tables: dict[str, pl.DataFrame],
) -> None:
    """Café con Jiribilla says 1 kg in the title and 250 g in the platform's field."""
    coarse = clean_tables["roaster_offers"].filter(
        pl.col("variant_title") == "Molido grueso / 1 kg"
    )

    assert coarse["bag_grams"].item() == 1000
    assert coarse["price_mxn_per_kg"].item() == coarse["price_mxn"].item()
