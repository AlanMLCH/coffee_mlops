"""How a roaster's sheet is read: every case here was found in the shops' real sheets."""

import logging
from datetime import UTC, date, datetime

import polars as pl
import pytest

import domains.coffee
from domains.coffee.roaster_sheets import (
    altitude_range,
    bag_grams,
    clean_roasters,
    place,
    processing_method,
    variety_list,
)
from domains.coffee.schemas import clean_schemas
from mlops_core.contracts import check_contract

RULES = domains.coffee.adapter().config.cleaning
READ_AT = datetime(2026, 9, 21, 23, 14, tzinfo=UTC)
SHEETS = RULES.roaster_sheets


@pytest.mark.parametrize(
    ("text", "country", "state"),
    [
        ("Etiopía", "Ethiopia", None),
        ("Tenejapa, Chiapas", None, "Chiapas"),
        ("Nuevo México, Chiapas", None, "Chiapas"),  # a town, not the country
        ("Atoyac Guerrero, México", "Mexico", None),  # a bare "México" is the country
        ("Estado de México", None, "México"),  # SIAP's name for the state
        ("Finca Manto Niebla, Cerro Espino, Pluma Hidalgo", None, None),  # not Hidalgo
    ],
)
def test_places_are_read_part_by_part(text: str, country: str | None, state: str | None) -> None:
    assert place(text, SHEETS) == (country, state)


@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("1,200 - 1,700 msnm", 1200, 1700),
        ("2,000  a 2,400 msnm", 2000, 2400),
        ("1,850 – 2,100 msnm", 1850, 2100),  # noqa: RUF001 - the en dash is the shop's
        ("1350 m.s.n.m.", 1350, 1350),
        ("1,1150 msnm", 1150, 1150),  # a typo: the stray 1 is not an altitude
        ("+2,000 msnm (cultivo), menor en estación de lavado", 2000, 2000),
        ("Alta montaña", None, None),
    ],
)
def test_altitudes_are_ranges_in_metres(text: str, low: float | None, high: float | None) -> None:
    assert altitude_range(text, RULES.altitude_m) == (low, high)


@pytest.mark.parametrize(
    ("text", "varieties"),
    [
        ("Typica, Bourbon y Marsellesa", ["bourbon", "marsellesa", "typica"]),
        (
            "85% Typica 15% Caturra, Bourbón, Sarchimor",
            ["bourbon", "caturra", "sarchimor", "typica"],
        ),
        ("SL-28, SL-34, Ruiru-11", ["ruiru 11", "sl28", "sl34"]),  # as the CQI spells them
        ("Hibridos de Timor y Tipicas.", ["timor hybrid", "typica"]),
        ("S-794 (Lini-S)", ["s-794"]),
        ("Heirloom Cultivars", ["heirloom"]),  # not the CQI's "ethiopian": this one is Yemeni
    ],
)
def test_varieties_are_split_and_spelled_as_the_cqi_does(text: str, varieties: list[str]) -> None:
    assert variety_list(text, SHEETS.varieties) == varieties


@pytest.mark.parametrize(
    ("label", "method"),
    [
        ("Lavado", "washed"),
        ("Lavado (Fully Washed)", "washed"),
        ("Natural (Seco)", "natural"),
        ("Enmielado Negro", "honey"),
        ("Semi-Lavado", "semi_washed"),  # not also "lavado"
        ("Giling Basah (Trillado Humedo)", "semi_washed"),
        ("Lavado, Natural", "other"),  # a lot sold two ways
        ("Natural Maceración Carbónica", "other"),  # as the CQI rules file experiments
        ("Hidronatural Lavado", "other"),
        ("Mycoprism", "unclassified"),  # recognised by nothing: counted, not guessed
    ],
)
def test_process_labels_speak_the_cqi_vocabulary(label: str, method: str) -> None:
    assert processing_method(label, SHEETS) == method


@pytest.mark.parametrize(
    ("variant", "title", "grams"),
    [
        ("5/16 kg (312.5 gr)", "Yemen Mokha Haimi", 312.5),
        ("Lavado Typica / 5/4 kg (1.25 kg)", "Don Federico", 1250),
        ("Natural Honey 280 horas / 5/16 kg (312.5 gr)", "Finca Corahe", 312.5),
        ("Molido grueso / 1 kg", "Mezcla Mística", 1000),
        ("1 Kg.", "Chiapas- Caramelo", 1000),
        ("Compra única / 340 gr / Molido medio", "Café Dalia", 340),
        ("Default Title", "Café Ikaria 250g", 250),  # the product's size
        ("En grano", "Caja de 12 Bolsas Café Dalia de 340grs", 4080),  # a pack
        ("DALIA MAUI OME .340G / EN GRANO", "Café / selección del tostador / 3 x 340g", 1020),
        ("KIT RICO / EN GRANO", "Kit Rico", None),  # no size anywhere: none invented
        (None, "Puebla - Durazno dulce", None),
    ],
)
def test_the_size_comes_from_the_titles(
    variant: str | None, title: str, grams: float | None
) -> None:
    assert bag_grams(variant, title) == grams


def offer(  # type: ignore[type-arg]
    product: str, variant: str, title: str, size: str, price: float, **extra: object
) -> dict:
    """One raw offer, as `roasters.to_frame` shapes it."""
    return {
        "shop": "shop",
        "platform": "shopify",
        "product_id": product,
        "variant_id": variant,
        "title": title,
        "variant_title": size,
        "price": price,
        "platform_grams": 0.0,
        "url": f"https://shop.test/products/{product}",
        "tags": None,
        "body_html": None,
        "page_html": None,
        **extra,
    }


def test_a_price_copied_from_another_size_is_flagged_not_fixed() -> None:
    """Almanegra lists a 156 g bag at the 1.25 kg price: kept as listed, flagged."""
    raw = pl.DataFrame(
        [
            offer("p", "small", "Korgua", "5/32 kg (156.25 gr)", 1350.0),
            offer("p", "medium", "Korgua", "5/16 kg (312.5 gr)", 349.0),
            offer("p", "large", "Korgua", "5/4 kg (1.25 kg)", 1350.0),
        ]
    )

    offers = clean_roasters(raw, RULES, READ_AT)["roaster_offers"]

    flagged = dict(offers.select("variant_id", "price_outlier").iter_rows())
    assert flagged == {"small": True, "medium": False, "large": False}
    assert offers.filter(pl.col("variant_id") == "small")["price_mxn"].item() == 1350.0


def test_a_kit_has_a_size_but_no_price_per_kilogram() -> None:
    """A kit's price pays for a book or chocolate as well as its coffee."""
    raw = pl.DataFrame([offer("k", "v", "Kit El monje de Moka y Café de Yemen", "5/32 kg", 599.0)])

    row = clean_roasters(raw, RULES, READ_AT)["roaster_offers"].row(0, named=True)

    assert row["bag_grams"] == 156.25
    assert row["price_mxn_per_kg"] is None
    assert row["price_outlier"] is None


def test_a_country_no_rule_maps_is_left_empty_and_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = "<p>País: Atlantis</p><p>Proceso: Lavado</p>"
    raw = pl.DataFrame([offer("a", "v", "Atlantis Lavado", "250 g", 300.0, body_html=body)])

    with caplog.at_level(logging.WARNING):
        origin = clean_roasters(raw, RULES, READ_AT)["roaster_origins"].row(0, named=True)

    assert origin["country"] is None and origin["processing_method"] == "washed"
    assert "Atlantis" in caplog.text


def test_shops_never_read_leave_empty_tables_that_keep_their_contracts() -> None:
    tables = clean_roasters(None, RULES)
    contracts = clean_schemas(RULES)

    for name, table in tables.items():
        assert table.is_empty()
        check_contract(contracts[name], table)


def test_offers_carry_one_column_keys_and_the_day_they_were_seen() -> None:
    """A platform's ids repeat across shops; the read date is what stage 4 compares."""
    raw = pl.DataFrame([offer("p", "v", "Korgua", "5/16 kg", 349.0)])

    tables = clean_roasters(raw, RULES, READ_AT)

    row = tables["roaster_offers"].row(0, named=True)
    assert (row["offer_id"], row["coffee_id"]) == ("shop-v", "shop-p")
    assert (row["observed_on"], row["snapshot"]) == (date(2026, 9, 21), "2026-09-21")
    assert tables["roaster_coffees"]["coffee_id"].to_list() == ["shop-p"]


def test_offers_without_the_time_they_were_read_are_refused() -> None:
    raw = pl.DataFrame([offer("p", "v", "Korgua", "5/16 kg", 349.0)])

    with pytest.raises(ValueError, match="read"):
        clean_roasters(raw, RULES)
