"""Mexico City's Metro and Metrobús: stations from the city's GTFS feed placed in their
boroughs, and the daily entries repaired, matched and set beside the coffee shops.

Read from the written files the whole suite uses (`tests.conftest.TRANSIT_STOPS`,
`METRO_RIDERSHIP`, `METROBUS_RIDERSHIP`).
"""

import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.analysis import borough_transit, figures, transit_by_year
from domains.coffee.config import CoffeeConfig
from domains.coffee.schemas import TRANSIT_RIDERSHIP, TRANSIT_STATIONS
from domains.coffee.transit import (
    clean_transit_ridership,
    clean_transit_stations,
    folded,
    line_code,
    repaired,
)
from mlops_core.contracts import check_contract
from mlops_core.data.extract import ingest
from mlops_core.data.validate import validate_read
from tests.conftest import BALDERAS_ZONE

Frames = dict[str, pl.DataFrame]


@pytest.fixture
def raw(coffee_adapter: CoffeeAdapter, client: Any, tmp_path: Path) -> Frames:
    """The three transit sources, the boroughs and their AGEBs, downloaded and checked as a
    build does."""
    config = coffee_adapter.config
    transit = config.transit
    names = [transit.stops, transit.metro, transit.metrobus, "cdmx_boroughs", "cdmx_ageb"]
    frames = {}
    for name in names:
        artifact = ingest(name, config.sources[name], tmp_path / "raw", client)
        frames[name] = validate_read(coffee_adapter, name, artifact).frame
    return frames


@pytest.fixture
def stations(raw: Frames) -> pl.DataFrame:
    return check_contract(
        TRANSIT_STATIONS,
        clean_transit_stations(raw["transit_stops"], raw["cdmx_boroughs"], raw["cdmx_ageb"]),
    )


def test_a_name_is_folded_to_its_words_and_a_broken_one_repaired() -> None:
    assert folded("Etiopía / Plaza de la Transparencia") == "etiopia plaza de la transparencia"
    assert folded("Ferrería y Arena Ciudad de México") == folded("Ferreria-Arena Ciudad de Mexico")
    assert repaired("PantitlÃ¡n") == "Pantitlán"
    assert repaired("Miguel Ãngel de Quevedo") == "Miguel Ángel de Quevedo"  # via Latin-1
    assert repaired("Balderas") == "Balderas"
    assert repaired("Pantitlán") == "Pantitlán"  # never broken: its accent stays
    assert [line_code(s) for s in ("Linea 1", "LÃ\xadnea 12", "linea b")] == ["1", "12", "B"]


def test_each_station_is_placed_once_per_line(stations: pl.DataFrame) -> None:
    metro = stations.filter(pl.col("system") == "metro")
    assert sorted(metro["line"]) == ["1", "1", "12", "2", "A"]
    # One Metrobús station for its two platforms, at their middle; the trolleybus is out.
    metrobus = stations.filter(pl.col("system") == "metrobus")
    assert sorted(metrobus["station"]) == ["20 de Noviembre", "Pino Suárez Sur"]
    noviembre = metrobus.filter(pl.col("station") == "20 de Noviembre").row(0, named=True)
    assert noviembre["latitude"] == pytest.approx(19.5051)
    assert noviembre["station_id"] == "metrobus-4-20-de-noviembre"
    # La Paz is over the city line: kept, with no borough.
    paz = stations.filter(pl.col("station") == "La Paz").row(0, named=True)
    assert paz["borough_id"] is None
    balderas = stations.filter(pl.col("station") == "Balderas").row(0, named=True)
    assert (balderas["borough"], balderas["zone_id"]) == ("Cuauhtémoc", BALDERAS_ZONE)


def test_the_metros_days_are_repaired_matched_and_resolved(
    raw: Frames,
    stations: pl.DataFrame,
    coffee_config: CoffeeConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)

    table = check_contract(
        TRANSIT_RIDERSHIP,
        clean_transit_ridership(
            raw["metro_ridership"], raw["metrobus_ridership"], stations, coffee_config.transit
        ),
    )

    metro = table.filter(pl.col("system") == "metro")
    # The double-encoded day is Pantitlán's, on line 1.
    broken_day = metro.filter(pl.col("date") == date(2022, 3, 1)).row(0, named=True)
    assert (broken_day["line"], broken_day["station"]) == ("1", "Pantitlán")
    # The longer name is the Zócalo; a closed day is zero, not nothing.
    assert metro.filter(pl.col("station") == "Zócalo")["entries"].to_list() == [30000]
    assert metro.filter(pl.col("station") == "Tláhuac")["entries"].to_list() == [0]
    # The day Balderas is named twice is not a day of it.
    assert metro.filter(pl.col("date") == date(2020, 12, 15)).is_empty()
    assert "2 rows name a station twice on one day (Balderas on line 1" in caplog.text
    # The Metrobús, per line: its two spellings of line 4 are one line; the day before it
    # opened is not a day of it.
    metrobus = table.filter(pl.col("system") == "metrobus")
    assert metrobus.select("line", "entries", "station_id").rows() == [
        ("4", 80000, None),
        ("4", 82000, None),
    ]


def test_a_station_the_feed_does_not_have_is_refused(
    raw: Frames, stations: pl.DataFrame, coffee_config: CoffeeConfig
) -> None:
    """A station the counts name and the feed lacks would have no place: refused, not
    dropped, so a new station or a new spelling is seen."""
    unknown = raw["metro_ridership"].with_columns(pl.lit("Estación Nueva").alias("estacion"))

    with pytest.raises(ValueError, match="name stations the feed does not have"):
        clean_transit_ridership(unknown, raw["metrobus_ridership"], stations, coffee_config.transit)


@pytest.fixture
def ridership(raw: Frames, stations: pl.DataFrame, coffee_config: CoffeeConfig) -> pl.DataFrame:
    return clean_transit_ridership(
        raw["metro_ridership"], raw["metrobus_ridership"], stations, coffee_config.transit
    )


def test_boroughs_weigh_their_metro_stations_by_their_open_days(
    stations: pl.DataFrame, ridership: pl.DataFrame
) -> None:
    boroughs = pl.DataFrame(
        {
            "borough_id": ["09015", "09011", "09002"],
            "borough": ["Cuauhtémoc", "Tláhuac", "Azcapotzalco"],
            "area_km2": [32.4, 85.0, 33.5],
            "population": [545884, 392313, 432205],
            "schooling_years": [12.4, 10.0, 11.0],
            "workplaces": [1, 1, 1],
            "jobs_estimate": [1.0, 1.0, 1.0],
        }
    )
    shops = pl.DataFrame(
        {
            "borough_id": ["09015", "09015"],
            "source": ["denue", "denue"],
            "kind": ["coffee", "juice"],
        }
    )

    table = borough_transit(stations, ridership, shops, boroughs)

    centre = table.filter(pl.col("borough_id") == "09015").row(0, named=True)
    # Balderas' two days and the Zócalo's one, each averaged on its own; two Metrobús.
    assert centre["metro_daily_entries"] == pytest.approx(21000 + 30000)
    assert (centre["metro_stations"], centre["metrobus_stations"]) == (2, 2)
    assert centre["coffee_shops"] == 1
    assert centre["metro_entries_per_resident"] == pytest.approx(51000 / 545884)
    # Tláhuac's station was closed every day counted: a station, no entries.
    tlahuac = table.filter(pl.col("borough_id") == "09011").row(0, named=True)
    assert (tlahuac["metro_stations"], tlahuac["metro_daily_entries"]) == (1, 0.0)
    azcapotzalco = table.filter(pl.col("borough_id") == "09002").row(0, named=True)
    assert (azcapotzalco["metro_stations"], azcapotzalco["coffee_shops"]) == (0, 0)


def test_each_system_has_an_average_day_a_year(
    ridership: pl.DataFrame, coffee_config: CoffeeConfig
) -> None:
    by_year = transit_by_year(ridership)

    metro_2026 = by_year.filter(pl.col("system") == "metro", pl.col("year") == 2026).row(
        0, named=True
    )
    # 30 July: 20,000 + 150,000 + 30,000 + 0 + 40,000; 31 July: 22,000 + 158,000.
    assert metro_2026["daily_entries"] == pytest.approx((240000 + 180000) / 2)
    assert metro_2026["days"] == 2
    columns = ("source", "scope", "per_10k_people", "schooling_years")
    nothing = pl.DataFrame(
        schema=dict.fromkeys(columns[:2], pl.String) | dict.fromkeys(columns[2:], pl.Float64)
    )
    tables = defaultdict(lambda: nothing, {"transit_by_year": by_year})
    assert list(figures(tables, coffee_config.market_analysis)) == ["transit_ridership"]
