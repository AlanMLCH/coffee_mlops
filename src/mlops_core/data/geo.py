"""Geospatial reading and point-in-area attribution, through DuckDB's spatial extension.

The extension is confined to this module. Layers stay plain Parquet and geometry travels
as WKB in WGS84, so features, the API, the analysis and the agent read an area table
without installing or loading anything spatial. Two jobs live here: turn a downloaded map
layer into a frame of areas, and say which area each point falls in.

Three traps, each of them silent, each found against the real data:

- **`ST_Transform` needs `always_xy := true`.** EPSG:4326 officially orders its axes
  latitude first, so without it the call does not fail -- it returns coordinates that
  are merely wrong: a city's districts came back on another continent.
- **A shapefile's DBF declares no character set.** A Latin-1 one read as
  UTF-8 raises "Invalid unicode" rather than mangling a few names, so the encoding has
  to be stated in the config.
- **Areas are measured in the layer's own projection**, in metres. Lambert Conformal
  Conic preserves angles, not areas, so the number carries a little distortion: the 16
  areas of one real layer sum to 1,486 km2 against the published 1,495 (0.6% out). That
  is the accuracy to expect from these figures, and it is plenty for a density.
"""

import logging
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from mlops_core.config import SpatialConfig

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"  # what every point source in this project speaks
AREA_COLUMNS = ["area_id", "area_name", "area_km2", "boundary"]


def spatial_connection() -> "duckdb.DuckDBPyConnection":
    """A DuckDB connection with `spatial` loaded, or a message that says what is missing.

    DuckDB is imported here, on use, not at the top of the module: validation and
    cleaning import this module, and an environment that never reads a map layer - the
    prediction API's - must be able to import them without installing a SQL engine.
    """
    import duckdb

    con = duckdb.connect()
    try:
        con.install_extension("spatial")
        con.load_extension("spatial")
    except duckdb.Error as unavailable:  # first use downloads it; offline it cannot
        con.close()
        raise RuntimeError(
            "DuckDB's `spatial` extension is required to read the geospatial sources. "
            "It downloads once and is cached in ~/.duckdb, so this needs network the "
            "first time."
        ) from unavailable
    return con


def read_areas(archive: Path, member: str, spatial: SpatialConfig) -> pl.DataFrame:
    """Read one map layer out of a ZIP into `area_id`, `area_name`, `area_km2`, `boundary`.

    The layer is read in place through GDAL's `/vsizip/`, so the 83 MB download is never
    unpacked, and the geometry comes out as WKB in WGS84: one shape per row, no
    dependency on this module to read it back.
    """
    layer = f"/vsizip/{archive.as_posix()}/{member}"
    # Column and option names come from the domain config, not from user input; they are
    # interpolated for the same reason the catalog interpolates table names.
    wgs84 = f"ST_Transform(geom, '{spatial.crs}', '{WGS84}', always_xy := true)"
    query = f"""
        SELECT {spatial.id_column} AS area_id,
               {spatial.name_column} AS area_name,
               ST_Area(geom) / 1e6 AS area_km2,
               ST_AsWKB({wgs84}) AS boundary
        FROM ST_Read(?, open_options = ['ENCODING={spatial.encoding}'])
        ORDER BY area_id
    """
    with closing(spatial_connection()) as con:
        areas: pl.DataFrame = con.execute(query, [layer]).pl()

    if areas.height != spatial.expected_features:
        raise ValueError(
            f"{member} has {areas.height} features, expected {spatial.expected_features}: "
            "the upstream boundary set changed, so check it before trusting the join"
        )
    logger.info("%s: %d areas, %.0f km2 in total", member, areas.height, areas["area_km2"].sum())
    return areas


def attribute_points(
    points: pl.DataFrame, areas: pl.DataFrame, latitude: str, longitude: str
) -> pl.DataFrame:
    """Add `area_id` and `area_name` to every point, null when it is outside them all.

    A left join on purpose: a point that falls outside every polygon is a fact about the
    data (a bad coordinate, or a place just over the city line), not a reason to lose the
    row. The count is the caller's to report.
    """
    if points.is_empty():  # DuckDB cannot infer the shape of an empty registration
        return points.with_columns(
            pl.lit(None, pl.String).alias("area_id"), pl.lit(None, pl.String).alias("area_name")
        )
    query = f"""
        SELECT p.*, a.area_id, a.area_name
        FROM points p
        LEFT JOIN areas a
          ON ST_Within(ST_Point(p."{longitude}", p."{latitude}"), ST_GeomFromWKB(a.boundary))
    """
    with closing(spatial_connection()) as con:
        con.register("points", points)
        con.register("areas", areas)
        attributed: pl.DataFrame = con.execute(query).pl()

    if attributed.height != points.height:
        # One point inside two areas duplicates its row and would inflate every count
        # downstream. Administrative areas do not overlap, so this means a broken layer.
        raise ValueError(
            f"The spatial join turned {points.height} points into {attributed.height} rows: "
            "the areas overlap"
        )
    return attributed
