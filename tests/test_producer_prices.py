"""FAOSTAT's producer prices of coffee: read out of the bulk file, cleaned to a year per
country, checked against SIAP where the figure is the cherry's, and set against the port.

Read from the written archive the whole suite uses (`tests.conftest.FAOSTAT_ROWS`).
"""

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from domains.coffee.analysis import farmgate_prices
from domains.coffee.config import CoffeeConfig, ProducerPricesConfig
from domains.coffee.prices import clean_producer_prices, reconcile_cherry
from domains.coffee.schemas import FAOSTAT_PRICES, PRODUCER_PRICES
from domains.coffee.sources.faostat import read_producer_prices
from mlops_core.contracts import check_contract
from tests.conftest import FAOSTAT_HEADER, FAOSTAT_MEMBER, FAOSTAT_ROWS, faostat_archive


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "prices.zip"
    path.write_bytes(faostat_archive())
    return path


@pytest.fixture
def farmers(coffee_config: CoffeeConfig) -> ProducerPricesConfig:
    return coffee_config.producer_prices


@pytest.fixture
def raw(archive: Path) -> pl.DataFrame:
    return check_contract(FAOSTAT_PRICES, read_producer_prices(archive, FAOSTAT_MEMBER, "656"))


def test_the_reader_keeps_only_the_item_even_when_its_code_is_another_crops_price(
    raw: pl.DataFrame,
) -> None:
    assert set(raw["Item"]) == {"Coffee, green"}
    assert raw.height == len(FAOSTAT_ROWS) - 1  # the almonds priced at 656 are gone


def test_a_file_that_is_not_faostats_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.zip"
    other.write_bytes(faostat_archive([], ["Region Id", *FAOSTAT_HEADER[1:]]))
    with pytest.raises(ValueError, match="FAOSTAT's header"):
        read_producer_prices(other, FAOSTAT_MEMBER, "656")


def test_a_column_the_contract_reads_that_is_gone_stops_the_read(tmp_path: Path) -> None:
    other = tmp_path / "other.zip"
    other.write_bytes(faostat_archive([], [*FAOSTAT_HEADER[:-1], "Note"]))
    with pytest.raises(ValueError, match=r"no \['Flag'\]"):
        read_producer_prices(other, FAOSTAT_MEMBER, "656")


def test_a_year_per_country_in_psds_names_with_zeros_and_months_left_out(
    raw: pl.DataFrame, farmers: ProducerPricesConfig
) -> None:
    table = check_contract(PRODUCER_PRICES, clean_producer_prices(raw, farmers))
    rows = {(r["country"], r["year"]): r for r in table.iter_rows(named=True)}
    assert set(rows) == {("Brazil", 2022), ("Colombia", 2023), ("Mexico", 2024), ("Vietnam", 2023)}
    brazil = rows["Brazil", 2022]
    # The year's value, not January's; the estimated zero is no index at all.
    assert (brazil["usd_per_t"], brazil["lcu_per_t"], brazil["price_index"]) == (
        3163.2, 16300.0, None
    )  # fmt: skip
    assert brazil["flag"] == "A"
    assert rows["Mexico", 2024]["cherry"] is True
    assert rows["Brazil", 2022]["cherry"] is False
    assert rows["Vietnam", 2023]["usd_per_t"] is None  # only its own currency


def test_the_same_code_under_another_name_is_refused(
    raw: pl.DataFrame, farmers: ProducerPricesConfig
) -> None:
    renamed = raw.with_columns(pl.lit("Coffee, roasted").alias("Item"))
    with pytest.raises(ValueError, match="not Coffee, green"):
        clean_producer_prices(renamed, farmers)


def test_a_year_with_only_the_index_keeps_the_other_columns(farmers: ProducerPricesConfig) -> None:
    only_index = pl.DataFrame(
        {
            "Area": ["Peru"],
            "Item": ["Coffee, green"],
            "Element Code": ["5539"],
            "Year": [2020],
            "Months": ["Annual value"],
            "Value": [120.0],
            "Flag": ["E"],
        }
    )
    table = check_contract(PRODUCER_PRICES, clean_producer_prices(only_index, farmers))
    assert table.select("usd_per_t", "lcu_per_t", "price_index", "flag").row(0) == (
        None, None, 120.0, "E"
    )  # fmt: skip


def mexico(year: int, value_mxn: float, production_t: float) -> dict[str, object]:
    return {"year": year, "value_mxn": value_mxn, "production_t": production_t}


def test_the_cherry_price_is_checked_against_the_farmers_own(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prices = pl.DataFrame(
        {"country": ["Mexico"] * 3, "year": [2022, 2023, 2024], "lcu_per_t": [100.0, 99.5, 150.0]}
    )
    production = pl.DataFrame([
        mexico(2022, 6_000, 40), mexico(2022, 4_000, 60),  # two municipalities: 10,000 / 100
        mexico(2023, 10_000, 100), mexico(2024, 10_000, 100),
    ])  # fmt: skip
    caplog.set_level(logging.INFO)
    assert reconcile_cherry(prices, production, "Mexico") == (2, 3)
    assert "the rural price of the cherry in 2 of 3 years" in caplog.text
    assert "WARNING" in caplog.text


def test_the_farmgate_is_set_against_the_port_price_of_its_own_mix() -> None:
    producer = pl.DataFrame(
        {
            "country": ["Brazil", "Mexico", "Sri Lanka"],
            "year": [2024] * 3,
            "usd_per_t": [3000.0, 300.0, 3000.0],
            "flag": ["A"] * 3,
            "cherry": [False, True, False],
        }
    )
    context = pl.DataFrame(
        {
            "country": ["Brazil", "Mexico", "Sri Lanka"],
            "market_year": [2024] * 3,
            "production": [100.0, 50.0, 0.0],
            "arabica_production": [75.0, 45.0, 0.0],
            "robusta_production": [25.0, 5.0, 0.0],
        }
    )
    # A year of months: 200 and 100 cents per pound, 4,409.2 and 2,204.6 dollars a tonne.
    months = [date(2024, m, 1) for m in range(1, 13)]
    indicators = pl.DataFrame(
        {
            "period": months * 3,
            "frequency": ["monthly"] * 24 + ["daily"] * 12,
            "indicator": ["other_milds"] * 12 + ["robustas"] * 12 + ["other_milds"] * 12,
            "usd_cents_per_lb": [200.0] * 12 + [100.0] * 12 + [900.0] * 12,
        }
    )
    table = farmgate_prices(producer, context, indicators)
    # The cherry is not compared, nor a country without a harvest to weigh by.
    assert table["country"].to_list() == ["Brazil"]
    row = table.row(0, named=True)
    assert row["arabica_share"] == 0.75
    assert row["benchmark_usd_per_t"] == pytest.approx(0.75 * 4409.245 + 0.25 * 2204.623, 1e-4)
    assert row["share_of_benchmark"] == pytest.approx(3000 / row["benchmark_usd_per_t"])
