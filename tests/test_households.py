"""INEGI's household survey (ENIGH): its coffee purchases read from 5 million rows, a row
per household with what it spent, and the survey's estimates with their intervals.

Read from the written archives the whole suite uses (`tests.conftest.ENIGH_*`).
"""

import zipfile
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from matplotlib.figure import Figure
from pydantic import ValidationError

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.analysis import household_coffee_figure, household_deciles_figure
from domains.coffee.config import CoffeeConfig, HouseholdSpendingConfig
from domains.coffee.households import (
    NATIONAL,
    clean_household_coffee,
    household_coffee_by_decile,
    household_coffee_by_state,
    survey_interval,
)
from domains.coffee.schemas import household_coffee_schema
from domains.coffee.sources.enigh import read_spending
from mlops_core.contracts import check_contract
from mlops_core.data.extract import ingest
from mlops_core.data.validate import validate_read
from tests.conftest import ENIGH_SPENDING


@pytest.fixture
def survey(coffee_config: CoffeeConfig) -> HouseholdSpendingConfig:
    return coffee_config.household_spending


@pytest.fixture
def raw(coffee_adapter: CoffeeAdapter, client: Any, tmp_path: Path) -> dict[str, pl.DataFrame]:
    """Both files, downloaded and validated as `mlops data validate` does."""
    config = coffee_adapter.config.household_spending
    frames = {}
    for name in (config.spending, config.households):
        artifact = ingest(name, coffee_adapter.config.sources[name], tmp_path / "raw", client)
        frames[name] = validate_read(coffee_adapter, name, artifact).frame
    return frames


@pytest.fixture
def table(raw: dict[str, pl.DataFrame], survey: HouseholdSpendingConfig) -> pl.DataFrame:
    return clean_household_coffee(raw[survey.spending], raw[survey.households], survey)


def test_only_the_coffee_purchases_are_read_and_a_blank_is_null(
    raw: dict[str, pl.DataFrame], survey: HouseholdSpendingConfig, tmp_path: Path
) -> None:
    spending = raw[survey.spending]

    assert set(spending["clave"]) == {"012201", "012202", "012203"}  # bread is not read
    assert spending.height == len(ENIGH_SPENDING) - 2  # the header and the bread
    grown = spending.filter(pl.col("tipo_gasto") == "G3").row(0, named=True)
    assert grown["gasto_tri"] is None and grown["gas_nm_tri"] == pytest.approx(257.14)
    # A file without a column the domain reads is another layout, said so.
    path = tmp_path / "changed.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("gastoshogar.csv", "folioviv,clave\n0900000101,012201\n")
    with pytest.raises(ValueError, match=r"has no \['foliohog', 'tipo_gasto'"):
        read_spending(path, "gastoshogar.csv", ["012201"])


def test_a_household_a_row_with_what_it_paid_and_what_it_grew(
    table: pl.DataFrame, survey: HouseholdSpendingConfig
) -> None:
    check_contract(household_coffee_schema(survey), table)
    rows = {row["household_id"]: row for row in table.iter_rows(named=True)}

    assert len(rows) == 8  # every household, with or without coffee
    city = rows["0900000101-1"]
    assert (city["state"], city["municipality_id"], city["weight"]) == ("Ciudad de México",
                                                                        "09015", 300)  # fmt: skip
    assert city["instant_quarter_mxn"] == pytest.approx(385.71)
    assert city["ground_quarter_mxn"] == pytest.approx(514.28)
    assert rows["0900000201-1"]["ground_quarter_mxn"] == pytest.approx(1285.71)  # two bags
    grower = rows["0700000101-1"]
    assert grower["instant_quarter_mxn"] == 0 and grower["own_harvest_quarter_mxn"] > 0
    gifted = rows["1500000101-2"]  # a gift is neither paid for nor grown
    assert gifted["prepared_quarter_mxn"] == 0 and gifted["own_harvest_quarter_mxn"] == 0
    assert rows["1500000101-2"]["state"] == "México"  # SIAP's name for the State of Mexico


def test_deciles_cut_the_households_by_the_households_they_stand_for(
    table: pl.DataFrame,
) -> None:
    """3,600 households in all: the poorest one (12,000 a quarter) stands for 200 of them,
    the first two tenths and a little more; the richest for the last 200."""
    deciles = dict(zip(table["household_id"], table["income_decile"], strict=True))

    assert deciles["0900000202-1"] == 1  # 200 of 3,600: the first tenth
    assert deciles["0700000101-1"] == 2  # the next 500 reach 700 of 3,600
    assert deciles["0900000201-1"] == 10  # the richest
    assert table["income_decile"].is_sorted() is False  # the table is by household


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s, h: (s, h.filter(pl.col("folioviv") != "0700000102")), "belong to no household"),
        (lambda s, h: (s.with_columns(entidad=pl.lit("21")), h), "name another state"),
        (lambda s, h: (s.with_columns(gasto_tri=None), h), "paid coffee purchases have no value"),
        (
            lambda s, h: (
                s.with_columns(entidad=pl.lit("99")),
                h.with_columns(ubica_geo=pl.lit("99001")),
            ),
            r"states \['99'\]",
        ),
    ],
)
def test_files_of_two_editions_do_not_make_a_table(
    raw: dict[str, pl.DataFrame], survey: HouseholdSpendingConfig, change: Any, message: str
) -> None:
    spending, households = change(raw[survey.spending], raw[survey.households])
    with pytest.raises(ValueError, match=message):
        clean_household_coffee(spending, households, survey)


def test_the_states_and_the_country_with_their_intervals(
    table: pl.DataFrame, survey: HouseholdSpendingConfig
) -> None:
    by_state = household_coffee_by_state(table, survey)
    rows = {row["state"]: row for row in by_state.iter_rows(named=True)}

    country = rows[NATIONAL]
    # 300 + 200 bought in the city (of 1,000) and 500 in Chiapas, of 3,600 in all.
    assert country["households"] == 3600
    assert country["bought_share"] == pytest.approx(1000 / 3600)
    city = rows["Ciudad de México"]
    assert city["bought_share"] == pytest.approx(0.5)
    assert city["monthly_mxn"] == pytest.approx((300 * 900.0 + 200 * 1285.71) / 1000 / 3, 1e-4)
    assert city["monthly_per_buyer_mxn"] == pytest.approx(city["monthly_mxn"] * 2)
    assert rows["Chiapas"]["own_harvest_share"] == pytest.approx(0.5)
    assert rows["México"]["monthly_per_buyer_mxn"] is None  # no one bought
    for row in rows.values():
        assert row["bought_share_low"] <= row["bought_share"] <= row["bought_share_high"]
        assert row["monthly_mxn_low"] <= row["monthly_mxn"] <= row["monthly_mxn_high"]


def test_deciles_for_the_country_and_the_city(
    table: pl.DataFrame, survey: HouseholdSpendingConfig
) -> None:
    by_decile = household_coffee_by_decile(table, survey)

    areas = by_decile["area"].unique(maintain_order=True).to_list()
    assert areas == [NATIONAL, "Ciudad de México"]
    city = by_decile.filter(area="Ciudad de México")
    assert city["households_sampled"].sum() == 4
    # The tenth decile holds the city's two richest households (90,000 and 150,000).
    richest = city.filter(income_decile=10).row(0, named=True)
    spent, earned = 300 * 900.0 + 200 * 1285.71, 300 * 90000 + 200 * 150000
    assert richest["income_per_mille"] == pytest.approx(1000 * spent / earned, 1e-4)


def test_an_interval_resamples_sampling_units_within_their_strata() -> None:
    """One unit in a stratum is always drawn: a stratum of one adds no spread."""
    one = pl.DataFrame({"stratum": ["a", "a"], "psu": ["1", "1"], "weight": [1, 3],
                        "bought": [1.0, 0.0]})  # fmt: skip
    assert survey_interval(one, "bought") == (0.25, 0.25)
    two = pl.DataFrame({"stratum": ["a", "a"], "psu": ["1", "2"], "weight": [1, 1],
                        "bought": [1.0, 0.0]})  # fmt: skip
    low, high = survey_interval(two, "bought")
    assert low == 0 and high == 1


def test_the_figures_set_the_city_apart(
    table: pl.DataFrame, survey: HouseholdSpendingConfig
) -> None:
    by_state = household_coffee_by_state(table, survey)
    assert isinstance(household_coffee_figure(by_state, "Ciudad de México"), Figure)
    assert isinstance(household_deciles_figure(household_coffee_by_decile(table, survey)), Figure)


def test_the_config_names_a_column_per_coffee_and_a_city_it_has(
    coffee_config: CoffeeConfig,
) -> None:
    survey = coffee_config.household_spending.model_dump()
    with pytest.raises(ValidationError, match="one column each"):
        HouseholdSpendingConfig(**survey | {"products": {"1": "Instant", "2": "ground"}})
    with pytest.raises(ValidationError, match="one column each"):
        HouseholdSpendingConfig(**survey | {"products": {"1": "ground", "2": "ground"}})
    with pytest.raises(ValidationError, match="is not in `states`"):
        HouseholdSpendingConfig(**survey | {"city": "33"})
    elsewhere = survey | {"households": "nowhere"}
    with pytest.raises(ValidationError, match="names the CSV inside INEGI's ZIP"):
        CoffeeConfig(**coffee_config.model_dump() | {"household_spending": elsewhere})
