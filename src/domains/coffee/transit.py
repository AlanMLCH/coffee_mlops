"""Mexico City's Metro and Metrobús -> `transit_stations` and `transit_ridership`.

Where people pass, beside where they live and where they work. The Metro counts the
entries at every station every day since 2010; the Metrobús counts them per line only,
since 2005, so its stations can be placed but not weighed. The stations come from the
city's GTFS feed, whose stop ids name the system and the line; each is placed in a
borough by the same spatial join the coffee shops go through.

Three things in the Metro's file, found reading it (29-30 September 2026):

- From January 2021 to May 2023 its lines and stations were written in UTF-8 read as
  Windows-1252 and encoded again ("LÃ­nea 1", "Isabel la CatÃ³lica"): undone on read,
  through Latin-1 where Windows-1252 has no character (an "Á" becomes "Ã" and a byte
  Windows-1252 leaves undefined).
- Two stations go by names the feed spells longer (the config's aliases); every other
  name of the sixteen years matches the feed, once accents and punctuation are dropped.
- In December 2020 line B lists "Oceanía" twice a day and "Deportivo Oceanía" not at all:
  the two rows cannot be told apart, so both are left out and the log says so.

A station closed for works reports zero entries, not nothing: kept as zero, so an average
over the days a station was open has to leave them out (the dictionary says so).
"""

import logging
import re
import unicodedata
from collections.abc import Mapping

import polars as pl

from domains.coffee.config import TransitConfig
from domains.coffee.schemas import METRO, METROBUS
from domains.coffee.zones import in_zones
from mlops_core.data.geo import attribute_points

logger = logging.getLogger(__name__)

# The feed's stop ids, verified 2026-09-30: "B_0200L1-PANTITLAN", "B_020L12-TLAHUAC" for
# the Metro, "B_0300L4-20NOVIEMBR" for the Metrobús. The line is in the id.
STOP_LINES = {
    METRO: r"^B_020(?:0L|L)(\w{1,2})[-_]",
    METROBUS: r"^B_0300L(\w)-",
}
FILLER = {"y", "linea"}  # words the two spellings of a name disagree on


def repaired(text: str) -> str:
    """UTF-8 read as Windows-1252 (or Latin-1) and encoded again, undone; a text that was
    never broken comes back as it was."""
    for codec in ("cp1252", "latin-1"):
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def folded(text: str) -> str:
    """A name reduced to its words: accents, case, punctuation and filler dropped, so
    "Etiopía / Plaza de la Transparencia" and "Etiopia-Plaza de la Transparencia" meet."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    words = re.sub(r"[^a-z0-9]+", " ", plain.lower()).split()
    return " ".join(w for w in words if w not in FILLER)


def line_code(text: str) -> str:
    """ "Línea 1", "LÃ­nea 12", "linea b" -> "1", "12", "B"."""
    return folded(repaired(text)).upper()


def clean_transit_stations(
    stops: pl.DataFrame, areas: pl.DataFrame, zones: pl.DataFrame
) -> pl.DataFrame:
    """A station per system, line and name, where the feed puts it, in a borough or none
    and in its urban AGEB.

    The Metrobús feed lists a platform per direction under one name: one station, at the
    middle of its platforms. A transfer is a station on each of its lines, as the Metro's
    own counts are.
    """
    systems = [
        stops.with_columns(
            pl.lit(system).alias("system"),
            pl.col("stop_id").str.extract(pattern).alias("line"),
        ).filter(pl.col("line").is_not_null())
        for system, pattern in STOP_LINES.items()
    ]
    named = pl.concat(systems).with_columns(
        pl.col("stop_name").map_elements(folded, return_dtype=pl.String).alias("key")
    )
    stations = (
        named.group_by("system", "line", "key", maintain_order=True)
        .agg(
            pl.col("stop_name").first().alias("station"),
            pl.col("stop_lat").mean().alias("latitude"),
            pl.col("stop_lon").mean().alias("longitude"),
        )
        .with_columns(
            pl.concat_str(
                "system", "line", pl.col("key").str.replace_all(" ", "-"), separator="-"
            ).alias("station_id")
        )
    )
    placed = attribute_points(stations, areas, "latitude", "longitude").rename(
        {"area_id": "borough_id", "area_name": "borough"}
    )
    outside = placed.filter(pl.col("borough_id").is_null())
    logger.info(
        "transit_stations: %d Metro and %d Metrobús stations; %d outside the city's "
        "boroughs (the State of Mexico)",
        placed.filter(pl.col("system") == METRO).height,
        placed.filter(pl.col("system") == METROBUS).height,
        outside.height,
    )
    return (
        in_zones(placed, zones)
        .select(
            "station_id", "system", "line", "station", "latitude", "longitude",
            "borough_id", "borough", "zone_id",
        )
        .sort("system", "line", "station")
    )  # fmt: skip


def clean_transit_ridership(
    metro: pl.DataFrame,
    metrobus: pl.DataFrame,
    stations: pl.DataFrame,
    transit: TransitConfig,
) -> pl.DataFrame:
    """Entries a day: per Metro station, per Metrobús line."""
    return pl.concat(
        [
            _metro_entries(metro, stations, transit.station_aliases),
            _metrobus_entries(metrobus),
        ]
    ).sort("date", "system", "line", "station_id", nulls_last=True)


def _metro_entries(
    metro: pl.DataFrame, stations: pl.DataFrame, aliases: Mapping[str, str]
) -> pl.DataFrame:
    names = (
        metro.select("linea", "estacion")
        .unique()
        .with_columns(
            pl.col("linea").map_elements(line_code, return_dtype=pl.String).alias("line"),
            pl.col("estacion")
            .map_elements(lambda s: folded(repaired(s)), return_dtype=pl.String)
            .replace(dict(aliases))
            .alias("key"),
        )
    )
    known = stations.filter(pl.col("system") == METRO).select(
        "line",
        pl.col("station").map_elements(folded, return_dtype=pl.String).alias("key"),
        "station_id",
        "station",
    )
    keyed = names.join(known, on=["line", "key"], how="left")
    missing = keyed.filter(pl.col("station_id").is_null())
    if not missing.is_empty():
        raise ValueError(
            "The Metro's counts name stations the feed does not have: "
            f"{sorted(zip(missing['line'], missing['key'], strict=True))}"
        )
    days = metro.join(keyed, on=["linea", "estacion"]).select(
        pl.col("fecha").str.to_date().alias("date"),
        pl.lit(METRO).alias("system"),
        "line",
        "station_id",
        "station",
        pl.col("afluencia").alias("entries"),
    )
    twice = days.filter(pl.struct("date", "station_id").is_duplicated())
    if not twice.is_empty():
        logger.warning(
            "metro: %d rows name a station twice on one day (%s on line %s, %s to %s); "
            "they cannot be told apart and are left out",
            twice.height,
            ", ".join(sorted(set(twice["station"]))),
            ", ".join(sorted(set(twice["line"]))),
            twice["date"].min(),
            twice["date"].max(),
        )
    return days.filter(~pl.struct("date", "station_id").is_duplicated())


def _metrobus_entries(metrobus: pl.DataFrame) -> pl.DataFrame:
    """A line's entries a day; the days before it opened ("NaN") are not days of it."""
    return metrobus.drop_nulls("afluencia").select(
        pl.col("fecha").str.to_date().alias("date"),
        pl.lit(METROBUS).alias("system"),
        pl.col("linea").map_elements(line_code, return_dtype=pl.String).alias("line"),
        pl.lit(None, pl.String).alias("station_id"),
        pl.lit(None, pl.String).alias("station"),
        pl.col("afluencia").cast(pl.Int64).alias("entries"),
    )
