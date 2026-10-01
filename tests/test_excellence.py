"""Cup of Excellence Mexico: a page a year read into its tables of lots and of sales, each
sale paired with its lot, and what a point of score is worth at auction.

Read from the written pages the whole suite uses (`tests.conftest.coe_page`).
"""

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.analysis import coe_figure
from domains.coffee.config import CoffeeConfig, RoasterSheetRules
from domains.coffee.excellence import clean_cup_of_excellence, coe_by_year, coe_score_price
from domains.coffee.schemas import cup_of_excellence_schema
from domains.coffee.sources.coe import read_competition
from mlops_core.contracts import check_contract
from mlops_core.data.extract import ingest, ingestions
from mlops_core.data.validate import validate_read
from tests.conftest import COE_JUDGES, coe_page, html_table

# SIAP's municipalities, by which a region naming only one is placed.
PRODUCTION = pl.DataFrame({"state": ["Chiapas", "Veracruz", "Puebla"],
                           "municipality": ["Ocosingo", "Coatepec", "Coatepec"]})  # fmt: skip


@pytest.fixture
def rules(coffee_config: CoffeeConfig) -> RoasterSheetRules:
    return coffee_config.cleaning.roaster_sheets


@pytest.fixture
def raw(coffee_adapter: CoffeeAdapter, client: Any, tmp_path: Path) -> pl.DataFrame:
    """Every year's page, downloaded and each validated as `mlops data validate` does."""
    source = coffee_adapter.config.sources["cup_of_excellence"]
    ingest("cup_of_excellence", source, tmp_path, client)
    years = ingestions(tmp_path, "cup_of_excellence")
    return pl.concat(validate_read(coffee_adapter, "cup_of_excellence", a).frame for a in years)


@pytest.fixture
def lots(raw: pl.DataFrame, rules: RoasterSheetRules) -> pl.DataFrame:
    return clean_cup_of_excellence(raw, rules, PRODUCTION)


def page(tmp_path: Path, year: int, body: str | bytes) -> Path:
    path = tmp_path / f"mexico-{year}.html"
    path.write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))
    return path


def test_a_page_is_read_as_its_tables_of_lots_and_sales(tmp_path: Path) -> None:
    rows = read_competition(page(tmp_path, 2026, coe_page(2026)))

    assert rows.height == 9  # the judges and the auction's totals are not lots
    assert set(rows["year"]) == {"2026"}
    winner = rows.filter(pl.col("farm") == "Finca Consuelo").row(0, named=True)
    assert (winner["process"], winner["variety"]) == ("NATURAL", "Borbón")  # one cell, split
    sold = read_competition(page(tmp_path, 2012, coe_page(2012))).filter(
        pl.col("bid").is_not_null()
    )
    assert sold["buyers"].to_list() == ["Maruyama", "Campos"]  # "High Bidder(s)" is not the bid
    with pytest.raises(ValueError, match="names no year"):
        read_competition(page(tmp_path, 2026, coe_page(2026)).rename(tmp_path / "results.html"))
    with pytest.raises(ValueError, match="no table of lots"):
        read_competition(page(tmp_path, 2027, "<html>" + COE_JUDGES + "</html>"))


def test_every_year_is_downloaded_but_the_ones_never_held(raw: pl.DataFrame) -> None:
    years = sorted(int(year) for year in raw["year"].unique())

    assert years == [2012, 2013, 2014, 2015, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025,
                     2026]  # fmt: skip


def test_each_sale_finds_its_lot(lots: pl.DataFrame, coffee_config: CoffeeConfig) -> None:
    check_contract(cup_of_excellence_schema(coffee_config.cleaning), lots)
    last = {row["farm"]: row for row in lots.filter(year=2026).iter_rows(named=True)}

    assert lots.height == 5 * 2 + 7 * 2 + 5 and lots["lot_id"].is_unique().all()
    # Before 2018, by lot number; its size in pounds is what it fetched over its price.
    old = lots.filter(year=2012, farm="Las Fincas Del Suspiro").row(0, named=True)
    assert old["price_usd_per_lb"] == 50.21 and old["weight_kg"] == pytest.approx(810.0, 0.01)
    assert (old["state"], old["processing_method"], old["varieties"]) == ("Veracruz", None, None)
    # Two first places in one year: rank and score tell them apart.
    assert last["Finca Santa Cruz"]["price_usd_per_lb"] == 92.0
    assert last["Pocitos"]["price_usd_per_lb"] == 40.7
    assert last["Pocitos"]["processing_method"] == "other"  # "Exerimental", as written
    # A decimal comma, and a farm the auction writes otherwise.
    viejo = next(row for farm, row in last.items() if farm.startswith("Rancho Viejo"))
    assert (viejo["score"], viejo["price_usd_per_lb"]) == (88.2, 14.6)
    assert viejo["varieties"] == ["bourbon", "typica"]
    # A national winner: no rank, its auction's score rounded; placed by its municipality.
    consuelo = last["Finca Consuelo"]
    assert consuelo["national_winner"] and consuelo["price_usd_per_lb"] == 4.0
    assert consuelo["state"] == "Chiapas"
    assert last["Unsold"]["price_usd_per_lb"] is None and last["Unsold"]["weight_kg"] == 200


def test_a_sale_is_never_guessed(tmp_path: Path, rules: RoasterSheetRules) -> None:
    header = ["Rank", "Farm", "Farmer", "Region", "Score"]
    alta, baja = ["1", "Alta", "A", "Puebla", "90.1"], ["2", "Baja", "B", "Puebla", "89"]
    results = html_table(header, alta, baja)
    twice = html_table(["Lot #", "Farm", "High Bid", "Total Value"],
                       ["1", "Alta", "$9/lb", "$900"], ["1", "Alta", "$8/lb", "$800"])  # fmt: skip
    nobody = html_table(["Lot #", "Farm", "High Bid", "Total Value"], ["7", "Otra", "$9/lb", "$9"])
    for sales, message in ((twice, "Two sales of one lot"), (nobody, "Sales of no lot")):
        raw = read_competition(page(tmp_path, 2030, results + sales))
        with pytest.raises(ValueError, match=message):
            clean_cup_of_excellence(raw, rules, PRODUCTION)
    # "Coatepec" is a municipality of two states: not placed.
    raw = read_competition(page(tmp_path, 2030, results.replace("Puebla", "Coatepec")))
    assert clean_cup_of_excellence(raw, rules, PRODUCTION)["state"].to_list() == [None, None]


def test_the_auction_against_the_market_year_by_year(lots: pl.DataFrame) -> None:
    indicators = pl.DataFrame({
        "period": [date(2026, 1, 1), date(2026, 2, 1), date(2026, 1, 1)],
        "frequency": ["monthly", "monthly", "monthly"],
        "indicator": ["other_milds", "other_milds", "robustas"],
        "usd_cents_per_lb": [300.0, 400.0, 100.0],
    })  # fmt: skip
    rates = pl.DataFrame(
        {"date": [date(2026, 1, 5), date(2026, 2, 5)], "mxn_per_usd": [17.0, 18.0]}
    )

    by_year = coe_by_year(lots, indicators, rates, "other_milds")
    last = by_year.filter(year=2026).row(0, named=True)

    assert (last["lots"], last["sold"], last["top_usd_per_lb"]) == (5, 4, 92.0)
    assert last["commodity_usd_per_lb"] == pytest.approx(3.5)
    median = float(np.median([92.0, 14.6, 40.7, 4.0]))
    assert last["median_times_commodity"] == pytest.approx(median / 3.5)
    assert last["median_mxn_per_kg"] == pytest.approx(median * 2.20462 * 17.5)
    assert by_year.filter(year=2012)["commodity_usd_per_lb"].item() is None


def test_a_point_of_score_is_worth_a_share_of_the_price() -> None:
    """Prices that rise 30% a point, each year from its own level: the slope says 30%."""
    rng = np.random.default_rng(0)
    score = rng.uniform(86, 92, 200)
    year = rng.choice([2024, 2025], 200)
    price = np.exp(np.log(1.3) * score + (year == 2025) * 0.5 - 100 + rng.normal(0, 0.01, 200))
    lots = pl.DataFrame({"lot_id": [f"{y}-{n:03d}" for n, y in enumerate(year)], "year": year,
                         "score": score, "price_usd_per_lb": price})  # fmt: skip

    worth = coe_score_price(lots, resamples=50).row(0, named=True)

    assert worth["percent_per_point"] == pytest.approx(30, abs=1)
    assert worth["percent_low"] < 30 < worth["percent_high"]
    assert worth["rank_correlation"] > 0.95


def test_the_figure_draws_the_years(lots: pl.DataFrame) -> None:
    columns = {"period": pl.Date, "frequency": pl.String, "indicator": pl.String}
    indicators = pl.DataFrame(schema=columns | {"usd_cents_per_lb": pl.Float64})
    rates = pl.DataFrame(schema={"date": pl.Date, "mxn_per_usd": pl.Float64})
    assert isinstance(coe_figure(coe_by_year(lots, indicators, rates, "other_milds")), Figure)


def test_a_row_of_both_tables_is_its_own_sale(tmp_path: Path, rules: RoasterSheetRules) -> None:
    """2017-2019 list their last lots in one table of results and auction; a heading may
    span a table, a table may be empty, and a lot with no region has no state."""
    both = html_table(
        ["Rank", "Farm", "Farmer", "Region", "Score", "High Bid", "Total Value"],
        ["Lots sold on their own"],
        ["29", "La Roca", "E. R.", "Coatepec, Veracruz", "85.63", "$4.00/lb", "$6,878.44"],
        ["30", "Sin Region", "F. L.", "", "85.50", "", ""],
    )
    raw = read_competition(page(tmp_path, 2018, "<table></table>" + both))

    lots = clean_cup_of_excellence(raw, rules, PRODUCTION)

    assert lots["price_usd_per_lb"].to_list() == [4.0, None]
    assert lots["state"].to_list() == ["Veracruz", None]
