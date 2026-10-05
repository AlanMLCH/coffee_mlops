"""INEGI's 2020 Census by urban AGEB: read down to each AGEB's own total, joined to the
framework's polygons, every place put in its zone, and the zones set side by side.

Read from the written files the whole suite uses (`tests.conftest.CENSUS_AGEB_ROWS` and
`framework_archive`, whose AGEBs are a grid over the boroughs' squares).
"""

import logging
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.analysis import (
    zone_coffee_correlations,
    zone_coffee_shops,
    zone_station_coffee,
)
from domains.coffee.schemas import CENSUS_ZONES
from domains.coffee.sources.census_zones import read_census_zones
from domains.coffee.zones import clean_census_zones, in_zones
from mlops_core.contracts import check_contract
from mlops_core.data.extract import ingest
from mlops_core.data.validate import validate_read
from tests.conftest import BALDERAS_ZONE, CENSUS_AGEB_MEMBER, ageb_code

Frames = dict[str, pl.DataFrame]


@pytest.fixture
def raw(coffee_adapter: CoffeeAdapter, client: Any, tmp_path: Path) -> Frames:
    config = coffee_adapter.config
    zones = config.census_zones
    frames = {}
    for name in (zones.census, zones.layer, "cdmx_boroughs"):
        artifact = ingest(name, config.sources[name], tmp_path / "raw", client)
        frames[name] = validate_read(coffee_adapter, name, artifact).frame
    return frames


@pytest.fixture
def zones(raw: Frames) -> pl.DataFrame:
    return check_contract(
        CENSUS_ZONES,
        clean_census_zones(raw["census_2020_ageb"], raw["cdmx_ageb"], raw["cdmx_boroughs"]),
    )


def test_only_each_agebs_own_total_is_read(raw: Frames) -> None:
    census = raw["census_2020_ageb"]

    assert census.height == 7  # no state, borough, locality or block rows
    assert set(census["NOM_LOC"]) == {"Total AGEB urbana"}
    assert "P_18YMAS" not in census.columns  # a column the zones do not use
    withheld = census.filter(pl.col("POBTOT") == 900)
    assert withheld["GRAPROES"].to_list() == [None]  # INEGI's "*"


def test_a_census_that_changed_its_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ageb.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(CENSUS_AGEB_MEMBER, "ENTIDAD,MUN,LOC,AGEB\n09,015,0001,2169\n")

    with pytest.raises(ValueError, match=r"has no .*'POBTOT'.*INEGI changed its layout"):
        read_census_zones(path, CENSUS_AGEB_MEMBER)


def test_each_zone_has_its_figures_its_polygon_and_its_borough(
    raw: Frames, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    table = check_contract(
        CENSUS_ZONES,
        clean_census_zones(raw["census_2020_ageb"], raw["cdmx_ageb"], raw["cdmx_boroughs"]),
    )

    # Six of the seven: the seventh the framework does not draw.
    assert table.height == 6
    assert "1 AGEBs the census counts have no polygon (75 people)" in caplog.text
    assert "2425 polygons have no census figures" in caplog.text
    balderas = table.filter(pl.col("zone_id") == BALDERAS_ZONE).row(0, named=True)
    assert (balderas["borough"], balderas["population"], balderas["dwellings"]) == (
        "Cuauhtémoc",
        3000,
        1000,
    )
    assert balderas["area_km2"] > 0
    # No AGEB has a name: it is read as INEGI writes its number, within its borough.
    number = f"{BALDERAS_ZONE[9:12]}-{BALDERAS_ZONE[12]}"
    assert (balderas["ageb"], balderas["label"]) == (number, f"Cuauhtémoc · AGEB {number}")
    # Where nobody lives, no average schooling - INEGI writes a 0.
    empty = table.filter(pl.col("population") == 0)
    assert empty["schooling_years"].to_list() == [None]


def test_a_point_is_placed_in_its_zone_or_none(raw: Frames) -> None:
    points = pl.DataFrame({"name": ["Balderas", "far"], "latitude": [19.50, 19.70],
                           "longitude": [-99.25, -99.25]})  # fmt: skip

    placed = in_zones(points, raw["cdmx_ageb"])

    assert placed["zone_id"].to_list() == [BALDERAS_ZONE, None]


# The zones' borough, for its jobs per resident.
BOROUGH = pl.DataFrame({"borough_id": ["09015"], "population": [100], "jobs_estimate": [50.0]})


def zone_rows(n: int) -> pl.DataFrame:
    """`n` zones of one km² each, the k-th with k people, k + 8 years of schooling and a
    k-th of its dwellings online."""
    return pl.DataFrame(
        {
            "zone_id": [f"z{k}" for k in range(n)],
            "borough_id": "09015",
            "borough": "Cuauhtémoc",
            "area_km2": [1.0] * n,
            "population": list(range(n)),
            "dwellings": [10] * n,
            "schooling_years": [k + 8.0 for k in range(n)],
            "economically_active": [1] * n,
            "people_65_plus": [0] * n,
            "dwellings_with_internet": list(range(n)),
            "dwellings_with_car": [1] * n,
            "dwellings_with_computer": [None] * n,  # withheld everywhere: no trait to rank
        },
        schema_overrides={"dwellings_with_computer": pl.Int64},
    )


def test_the_zones_set_coffee_shops_beside_what_the_census_counted() -> None:
    zones = zone_rows(8)
    shops = pl.DataFrame(
        {
            "zone_id": [f"z{k}" for k in range(8) for _ in range(k)] + ["z7"],
            "source": "denue",
            "kind": ["coffee"] * 28 + ["juice"],
        }
    )
    stations = pl.DataFrame(
        {"zone_id": ["z7", "z6", "z6"], "system": ["metro", "metro", "metrobus"]}
    )

    table = zone_coffee_shops(zones, shops, stations, BOROUGH)

    top = table.row(0, named=True)
    assert (top["zone_id"], top["coffee_shops"], top["metro_stations"]) == ("z7", 7, 1)
    assert top["internet_pct"] == pytest.approx(70.0)
    assert top["coffee_shops_per_km2"] == 7.0
    correlations = zone_coffee_correlations(table)
    assert "computer_pct" not in correlations["trait"].to_list()
    schooling = correlations.filter(pl.col("trait") == "schooling_years").row(0, named=True)
    assert schooling["rho"] == pytest.approx(1.0)
    assert schooling["zones"] == 8
    stations_rows = zone_station_coffee(table)
    metro = stations_rows.filter(pl.col("system") == "metro").row(0, named=True)
    assert (metro["zones_with"], metro["mean_with"]) == (2, 6.5)
    assert metro["difference"] == pytest.approx(6.5 - 15 / 6)
    assert metro["difference_low"] <= metro["difference"] <= metro["difference_high"]


def test_a_system_with_no_station_in_any_zone_is_not_compared() -> None:
    table = zone_coffee_shops(
        zone_rows(6),
        pl.DataFrame(schema={"zone_id": pl.String, "source": pl.String, "kind": pl.String}),
        pl.DataFrame({"zone_id": ["z1"], "system": ["metrobus"]}),
        BOROUGH,
    )

    assert zone_station_coffee(table)["system"].to_list() == ["metrobus"]
    assert zone_coffee_correlations(table.head(3)).is_empty()  # too few to rank
    assert zone_coffee_correlations(table).is_empty()  # no coffee shop anywhere: no order


def test_the_framework_draws_its_agebs_around_the_points_it_is_asked_about() -> None:
    assert ageb_code(19.50, -99.25) == BALDERAS_ZONE == "0901500012169"
