"""INEGI's 2020 Census by urban AGEB -> `census_zones`, and every place in its zone.

The borough is a coarse zone: sixteen of them, and the centre is one. The census also
counts every urban AGEB (a basic geostatistical area of a few dozen blocks), and the same
geostatistical framework that draws the boroughs draws the AGEBs. A coffee shop and a
station are placed in one the way they are placed in a borough, so what the boroughs
allow - coffee shops against residents, schooling, a station - can be set side by side
2,431 times instead of 16, and without a radius: the zone is the AGEB.

An AGEB the census counts and the framework does not draw, or the other way round, is
said in the log and left out of the table: a zone needs both its figures and its shape.
"""

import logging

import polars as pl

from mlops_core.data.geo import attribute_points

logger = logging.getLogger(__name__)

# The census' column -> the zone's.
ZONE = {
    "POBTOT": "population",
    "TVIVHAB": "dwellings",  # private dwellings lived in
    "GRAPROES": "schooling_years",
    "PEA": "economically_active",
    "POB65_MAS": "people_65_plus",
    "VPH_INTER": "dwellings_with_internet",
    "VPH_AUTOM": "dwellings_with_car",
    "VPH_PC": "dwellings_with_computer",
}


def clean_census_zones(
    census: pl.DataFrame, layer: pl.DataFrame, areas: pl.DataFrame
) -> pl.DataFrame:
    """Every urban AGEB with both its census figures and its polygon, in its borough."""
    figures = census.select(
        pl.concat_str("ENTIDAD", "MUN", "LOC", "AGEB").alias("zone_id"),
        *[pl.col(column).alias(name) for column, name in ZONE.items()],
    ).with_columns(
        # An AGEB nobody lives in (an airport, a park, an industrial estate) is written a
        # schooling of 0: an average of no one is no figure.
        pl.when(pl.col("population") > 0).then(pl.col("schooling_years")).alias("schooling_years")
    )
    polygons = layer.select(pl.col("area_id").alias("zone_id"), "area_km2", "boundary")
    unshaped = figures.join(polygons, on="zone_id", how="anti")
    uncounted = polygons.join(figures, on="zone_id", how="anti")
    if not unshaped.is_empty() or not uncounted.is_empty():
        logger.warning(
            "census_zones: %d AGEBs the census counts have no polygon (%d people), and %d "
            "polygons have no census figures; both left out",
            unshaped.height,
            unshaped["population"].sum(),
            uncounted.height,
        )
    names = areas.select(
        pl.col("area_id").alias("borough_id"), pl.col("area_name").alias("borough")
    )
    zones = (
        polygons.join(figures, on="zone_id")
        .with_columns(pl.col("zone_id").str.slice(0, 5).alias("borough_id"))
        .join(names, on="borough_id", how="left")
        .select("zone_id", "borough_id", "borough", "area_km2", "boundary", *ZONE.values())
        .sort("zone_id")
    )
    logger.info(
        "census_zones: %d urban AGEBs, %s people", zones.height, f"{zones['population'].sum():,}"
    )
    return zones


def in_zones(points: pl.DataFrame, layer: pl.DataFrame) -> pl.DataFrame:
    """Each point with the urban AGEB it falls in (`zone_id`), null outside every one."""
    shapes = layer.select("area_id", "area_name", "boundary")
    placed = attribute_points(points, shapes, "latitude", "longitude")
    return placed.rename({"area_id": "zone_id"}).drop("area_name")
