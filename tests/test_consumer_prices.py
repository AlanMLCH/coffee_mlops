"""PROFECO's shelf prices, per kilogram and in the borough their store declares.

Read from the written archive the whole suite uses (`tests.conftest.QQP_FORTNIGHTS`)
and the boundary fixture, whose 4x4 grid puts Polanco's store in Miguel Hidalgo and
Contreras' in La Magdalena Contreras.
"""

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from domains.coffee.config import CoffeeConfig, ConsumerPricesConfig
from domains.coffee.consumer_prices import clean_consumer_prices, restore_lost_letters
from domains.coffee.schemas import CONSUMER_PRICES, PROFECO_PRICES
from domains.coffee.sources.profeco import read_shelf_prices
from mlops_core.config import SpatialConfig
from mlops_core.contracts import check_contract
from mlops_core.data.geo import read_areas
from tests.conftest import GROUND, INSTANT, QQP_FORTNIGHTS, qqp_archive

BOUNDARIES = Path(__file__).parent / "fixtures" / "cdmx_boroughs_sample.zip"
LAYER = SpatialConfig(
    encoding="ISO-8859-1",
    crs="EPSG:6372",
    id_column="CVEGEO",
    name_column="NOMGEO",
    expected_features=16,
)


@pytest.fixture(scope="module")
def areas() -> pl.DataFrame:
    return read_areas(BOUNDARIES, "conjunto_de_datos/09mun.shp", LAYER)


@pytest.fixture
def raw(tmp_path: Path) -> pl.DataFrame:
    path = tmp_path / "QQP_2026.zip"
    path.write_bytes(qqp_archive(QQP_FORTNIGHTS))
    return check_contract(PROFECO_PRICES, read_shelf_prices(path, "QQP_2026", [INSTANT, GROUND]))


@pytest.fixture
def rules(coffee_config: CoffeeConfig) -> ConsumerPricesConfig:
    return coffee_config.consumer_prices


@pytest.fixture
def prices(raw: pl.DataFrame, areas: pl.DataFrame, rules: ConsumerPricesConfig) -> pl.DataFrame:
    return check_contract(CONSUMER_PRICES, clean_consumer_prices(raw, areas, rules))


def row(prices: pl.DataFrame, store: str, day: date) -> dict[str, object]:
    found = prices.filter((pl.col("store").str.contains(store)) & (pl.col("date") == day))
    assert found.height == 1, found
    return found.row(0, named=True)


def test_a_price_is_put_per_kilogram_with_what_its_presentation_declares(
    prices: pl.DataFrame,
) -> None:
    jar = row(prices, "Polanco", date(2026, 7, 20))
    decaf = row(prices, "Xalapa", date(2026, 5, 12))
    blend = prices.filter(pl.col("brand") == "Legal").row(0, named=True)

    assert (jar["product"], jar["grams"], jar["price_mxn_per_kg"]) == ("instant", 200.0, 950.0)
    assert (jar["sweetened"], jar["decaf"]) == (False, False)
    assert (decaf["decaf"], decaf["price_mxn_per_kg"]) == (True, 941.18)
    # Coffee and caramel, priced per kilogram of both.
    assert (blend["product"], blend["sweetened"], blend["price_mxn_per_kg"]) == (
        "ground", True, 239.75,
    )  # fmt: skip


def test_each_price_is_filed_under_its_fortnight(prices: pl.DataFrame) -> None:
    firsts = dict(zip(prices["date"], prices["fortnight"], strict=True))

    assert firsts[date(2026, 5, 4)] == date(2026, 5, 1)
    assert firsts[date(2026, 6, 5)] == date(2026, 6, 1)
    assert firsts[date(2026, 7, 21)] == date(2026, 7, 16)


def test_lost_letters_come_back_and_a_row_they_duplicated_is_kept_once(
    prices: pl.DataFrame,
) -> None:
    june = prices.filter(pl.col("date") == date(2026, 6, 3))

    assert june["brand"].to_list() == ["Nescafé. Clásico"]  # two rows before, one price
    assert not prices["brand"].str.contains("?", literal=True).any()


def test_two_prices_on_one_shelf_on_one_day_are_both_kept(
    raw: pl.DataFrame,
    areas: pl.DataFrame,
    rules: ConsumerPricesConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    prices = clean_consumer_prices(raw, areas, rules)

    same_day = prices.filter(pl.col("date") == date(2026, 6, 5))

    assert same_day["price_mxn"].to_list() == [160.0, 162.0]
    assert prices.height == raw.height - 1  # only the true duplicate went
    assert "1 times a shelf had two prices for one product on one day" in caplog.text


def test_the_borough_is_the_one_the_store_declares_in_inegis_spelling(
    raw: pl.DataFrame,
    areas: pl.DataFrame,
    rules: ConsumerPricesConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prices = clean_consumer_prices(raw, areas, rules)

    boroughs = dict(zip(prices["store"], prices["borough"], strict=False))

    assert boroughs["Walmart Sucursal Polanco"] == "Miguel Hidalgo"
    # PROFECO's "Magdalena Contreras" is INEGI's "La Magdalena Contreras".
    assert boroughs["Soriana Super Sucursal Contreras"] == "La Magdalena Contreras"
    # The market says Miguel Hidalgo and its coordinates say otherwise: it keeps its word,
    # and the log says so.
    assert boroughs["Mercado Tacuba"] == "Miguel Hidalgo"
    assert boroughs["Chedraui Sucursal Xalapa"] is None  # not in the city
    assert "coordinates fall in the declared borough for 2 of 3 stores" in caplog.text


def test_a_borough_name_inegi_does_not_have_is_said(
    raw: pl.DataFrame,
    areas: pl.DataFrame,
    rules: ConsumerPricesConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    unaliased = rules.model_copy(update={"borough_aliases": {}})

    prices = clean_consumer_prices(raw, areas, unaliased)

    assert "name no borough INEGI has: ['Magdalena Contreras']" in caplog.text
    assert prices.filter(pl.col("municipality") == "Magdalena Contreras")["borough"].is_null().all()


def test_a_store_elsewhere_whose_coordinates_are_in_the_city_is_said(
    raw: pl.DataFrame,
    areas: pl.DataFrame,
    rules: ConsumerPricesConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    misplaced = raw.with_columns(
        pl.when(pl.col("estado") == "Veracruz")
        .then(pl.lit(19.45))
        .otherwise(pl.col("latitud"))
        .alias("latitud"),
        pl.when(pl.col("estado") == "Veracruz")
        .then(pl.lit(-99.15))
        .otherwise(pl.col("longitud"))
        .alias("longitud"),
    )

    prices = clean_consumer_prices(misplaced, areas, rules)

    assert "1 stores outside Ciudad de México have coordinates inside it (2 prices)" in caplog.text
    assert prices.filter(pl.col("state") == "Veracruz")["borough"].is_null().all()


def test_a_letter_is_restored_only_where_one_whole_spelling_fits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    values = pl.Series(
        "municipio",
        ["Coyoac?n", "Coyoacán", "Le?n", "León", "Leún", "Tultitl?n", None],
    )

    restored = restore_lost_letters(values)

    assert restored.to_list() == ["Coyoacán", "Coyoacán", "Le?n", "León", "Leún", "Tultitl?n", None]
    assert "municipio: restored 1 values with lost letters; 2 have no single" in caplog.text
    clean = pl.Series("marca", ["Legal"])
    assert restore_lost_letters(clean) is clean  # nothing lost, nothing done


def test_each_fortnight_comes_from_the_latest_read_that_carries_it(
    raw: pl.DataFrame, areas: pl.DataFrame, rules: ConsumerPricesConfig
) -> None:
    """A new year's archive no longer holds the old year: its fortnights must stay. And a
    fortnight read twice keeps its later reading, since a correction can only come later."""
    from datetime import UTC, datetime

    may = pl.col("file") == "QQP_2026/05-2026_Q1.csv"
    first = raw.with_columns(pl.lit(datetime(2026, 8, 1, tzinfo=UTC)).alias("ingested_at"))
    # The next archive: May corrected (every price a peso more), and nothing else of 2026.
    later = raw.filter(may).with_columns(
        pl.col("precio") + 1, pl.lit(datetime(2027, 1, 16, tzinfo=UTC)).alias("ingested_at")
    )

    prices = clean_consumer_prices(pl.concat([first, later]), areas, rules)

    assert prices.height == clean_consumer_prices(raw, areas, rules).height  # nothing lost
    corrected = prices.filter(pl.col("date") == date(2026, 5, 4)).sort("price_mxn")
    assert corrected["price_mxn"].to_list() == [96.9, 111.0]
