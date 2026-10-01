"""INEGI's 2025 Intercensal Survey: read in its own encoding, cleaned to a row per borough
and indicator with its interval, checked against the city's totals, and set beside the
coffee shops.

Read from the written archive the whole suite uses (`tests.conftest.SURVEY_AREAS`).
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandera.errors
import polars as pl
import pytest
from matplotlib.figure import Figure

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.analysis import (
    MIN_BOROUGHS,
    borough_profile_coffee,
    coffee_and_profile_figure,
    figures,
)
from domains.coffee.config import BoroughProfileConfig, CoffeeConfig
from domains.coffee.schemas import borough_profile_schema
from domains.coffee.survey import clean_borough_profile
from mlops_core.contracts import check_contract
from mlops_core.data.extract import ingest
from mlops_core.data.validate import validate_read
from tests.conftest import SURVEY_INDICATORS

# Three boroughs the survey names, and one it does not.
AREAS = pl.DataFrame(
    {
        "area_id": ["09008", "09015", "09016", "09002"],
        "area_name": ["La Magdalena Contreras", "Cuauhtémoc", "Miguel Hidalgo", "Azcapotzalco"],
    }
)


@pytest.fixture
def profile(coffee_config: CoffeeConfig) -> BoroughProfileConfig:
    return coffee_config.borough_profile


@pytest.fixture
def survey(coffee_adapter: CoffeeAdapter, client: Any, tmp_path: Path) -> pl.DataFrame:
    name = coffee_adapter.config.borough_profile.source
    artifact = ingest(name, coffee_adapter.config.sources[name], tmp_path / "raw", client)
    return validate_read(coffee_adapter, name, artifact).frame


def test_the_config_reads_columns_the_survey_has(profile: BoroughProfileConfig) -> None:
    """A misspelt column would be a column the download lacks; the written archive uses
    the names the real file prints."""
    assert set(profile.indicators) <= set(SURVEY_INDICATORS)


def test_the_survey_is_read_in_its_own_encoding_with_a_thin_sample_as_null(
    survey: pl.DataFrame,
) -> None:
    assert survey.height == 30  # six areas, five estimators each
    assert "Límite inferior de confianza" in survey["ESTIMADOR"].to_list()
    thin = survey.filter(pl.col("CVE_MUN") == "008")["PCN_PRESOE20"]
    assert thin.null_count() == 5
    assert survey["POBTOT"].dtype == pl.Float64


def test_each_borough_has_a_row_per_indicator_inside_its_interval(
    survey: pl.DataFrame, profile: BoroughProfileConfig
) -> None:
    table = check_contract(
        borough_profile_schema(profile), clean_borough_profile(survey, AREAS, profile)
    )

    assert table.height == 3 * len(profile.indicators)  # no locality, no other state
    rented = table.filter(pl.col("borough_id") == "09015", pl.col("indicator") == "rented_pct")
    assert rented.row(0, named=True) | {"value": 0} == {
        "borough_id": "09015",
        "borough": "Cuauhtémoc",
        "year": 2025,
        "indicator": "rented_pct",
        "unit": "percent",
        "value": 0,
        "ci_low": pytest.approx(42.15 * 0.95),
        "ci_high": pytest.approx(42.15 * 1.05),
        "cv": 3.0,
    }
    assert rented["value"].item() == pytest.approx(42.15)
    moved = table.filter(
        pl.col("borough_id") == "09008", pl.col("indicator") == "lived_in_another_state_2020_pct"
    )
    assert moved["value"].to_list() == [None]  # INEGI's "MI": too few sampled


def test_a_borough_the_survey_does_not_name_is_said(
    survey: pl.DataFrame, profile: BoroughProfileConfig, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    table = clean_borough_profile(survey, AREAS, profile)

    assert "09002" not in table["borough_id"].to_list()
    assert "the survey names 3 of 4 boroughs" in caplog.text
    assert "estimates above a 15% coefficient of variation" in caplog.text


def test_counts_that_miss_the_citys_total_are_refused(
    survey: pl.DataFrame, profile: BoroughProfileConfig
) -> None:
    """A borough missing or misread, and the counts no longer come to the state's own."""
    short = survey.filter(pl.col("CVE_MUN") != "016")

    with pytest.raises(ValueError, match="the boroughs' population add up to 755,637"):
        clean_borough_profile(short, AREAS, profile)


def test_an_estimate_outside_its_own_interval_breaks_the_contract(
    survey: pl.DataFrame, profile: BoroughProfileConfig
) -> None:
    table = clean_borough_profile(survey, AREAS, profile)
    broken = table.with_columns(pl.col("value") * 2)

    with pytest.raises(pandera.errors.SchemaErrors, match="inside its own interval"):
        check_contract(borough_profile_schema(profile), broken)


def test_nothing_to_profile_is_an_empty_table(
    survey: pl.DataFrame, profile: BoroughProfileConfig
) -> None:
    elsewhere = pl.DataFrame({"area_id": ["31050"], "area_name": ["Mérida"]})

    assert clean_borough_profile(survey, elsewhere, profile).is_empty()


def places(boroughs: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`boroughs` of equal area and people, the n-th with n coffee shops."""
    ids = [f"09{n:03d}" for n in range(boroughs)]
    table = pl.DataFrame(
        {
            "borough_id": ids,
            "borough": [f"Borough {n}" for n in range(boroughs)],
            "area_km2": [10.0] * boroughs,
            "population": [100_000] * boroughs,
            "schooling_years": [12.0] * boroughs,
            "workplaces": [1000] * boroughs,
            "jobs_estimate": [5000.0] * boroughs,
        }
    )
    shops = pl.DataFrame(
        {
            "borough_id": [i for n, i in enumerate(ids) for _ in range(n + 1)],
            "source": "denue",
            "kind": "coffee",
        }
    )
    return table, shops


def survey_rows(indicator: str, values: list[float | None], unit: str = "percent") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "borough_id": [f"09{n:03d}" for n in range(len(values))],
            "borough": [f"Borough {n}" for n in range(len(values))],
            "indicator": indicator,
            "unit": unit,
            "value": values,
            "cv": [4.0] * len(values),
        }
    )


def test_the_study_ranks_each_indicator_against_coffee_shops_per_resident() -> None:
    boroughs, shops = places(8)
    rising, falling = [float(n) for n in range(8)], [float(8 - n) for n in range(8)]
    profile = pl.concat(
        [
            survey_rows("rented_pct", rising),
            survey_rows("occupants_per_room", falling, "people per room"),
            survey_rows("scrambled", [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]),
            # Known in too few boroughs to rank: left out, not reported as noise.
            survey_rows("thin", [1.0, 2.0, None, None, None, None, None, None]),
            # The same in every borough: no order to rank.
            survey_rows("flat", [5.0] * 8),
        ]
    )

    study = borough_profile_coffee(profile, shops, boroughs)

    assert study["indicator"].to_list()[:2] in (
        ["rented_pct", "occupants_per_room"],
        ["occupants_per_room", "rented_pct"],
    )
    assert {"thin", "flat"}.isdisjoint(study["indicator"].to_list())
    rented = study.filter(pl.col("indicator") == "rented_pct").row(0, named=True)
    assert rented["rho_per_10k_people"] == pytest.approx(1.0)
    assert rented["rho_low"] > 0.5
    assert rented["lowest"] == "Borough 0 0" and rented["highest"] == "Borough 7 7"
    falling_row = study.filter(pl.col("indicator") == "occupants_per_room").row(0, named=True)
    assert falling_row["rho_per_10k_people"] == pytest.approx(-1.0)
    assert falling_row["rho_per_km2"] == pytest.approx(-1.0)
    assert falling_row["loosest_cv"] == 4.0
    assert isinstance(coffee_and_profile_figure(study), Figure)


def test_a_count_reads_with_its_thousands() -> None:
    boroughs, shops = places(MIN_BOROUGHS)
    people = survey_rows("population", [1500.0, 2_000_000.0, 3000.0, 4000.0, 5000.0], "people")

    study = borough_profile_coffee(people, shops, boroughs)

    assert study["highest"].item() == "Borough 1 2,000,000"
    assert study["lowest"].item() == "Borough 0 1,500"


def test_the_figure_is_drawn_once_the_study_has_rows(coffee_config: CoffeeConfig) -> None:
    """The fixtures' census names too few boroughs for the study; on its own it draws."""
    boroughs, shops = places(MIN_BOROUGHS)
    study = borough_profile_coffee(
        survey_rows("rented_pct", [float(n) for n in range(MIN_BOROUGHS)]), shops, boroughs
    )
    nothing = pl.DataFrame(
        schema={"source": pl.String, "scope": pl.String,
                "per_10k_people": pl.Float64, "schooling_years": pl.Float64}
    )  # fmt: skip
    tables = defaultdict(lambda: nothing, {"borough_profile_coffee": study})

    drawn = figures(tables, coffee_config.market_analysis, "Ciudad de México")

    assert list(drawn) == ["coffee_and_profile"]
