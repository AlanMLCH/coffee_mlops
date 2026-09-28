"""The explorer's data: the domain's areas and map layers, read through the agent's locked
session - the same single read-only SELECT over the published layers that a question's
SQL runs in, so the app can show nothing a question could not ask for."""

import logging

import duckdb
import polars as pl

from mlops_core.agent.sql import run_select
from mlops_core.config import AreasConfig, ExploreConfig, MapLayer
from mlops_core.explore.charts import Areas, frame
from mlops_core.explore.shapes import feature_collection

logger = logging.getLogger(__name__)

MAP_ROWS = 20_000  # a layer is drawn, not read by a model: a city's places fit
AREA_ROWS = 1_000


def load_areas(con: duckdb.DuckDBPyConnection, areas: AreasConfig) -> Areas:
    """The domain's areas: the columns results name them by, and their outlines."""
    columns = ", ".join(f'"{column}"' for column in (areas.id, areas.name, areas.boundary))
    result = run_select(con, f"SELECT {columns} FROM {areas.table}", max_rows=AREA_ROWS)
    return Areas(areas.id, areas.name, feature_collection(result.rows))


def areas_if_built(con: duckdb.DuckDBPyConnection, explore: ExploreConfig | None) -> Areas | None:
    """The domain's areas, or None - it declares none, or their table is not built yet: a
    map of places still draws, only a map of areas cannot."""
    if explore is None or explore.areas is None:
        return None
    try:
        return load_areas(con, explore.areas)
    except duckdb.CatalogException:
        logger.warning("%s is not built yet: no map of areas until it is", explore.areas.table)
        return None


def run_layer(
    con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = MAP_ROWS
) -> tuple[pl.DataFrame, bool]:
    """A query's rows as a frame, and whether there were more than `max_rows`."""
    result = run_select(con, sql, max_rows=max_rows)
    return frame(result.columns, result.rows), result.truncated


def layer_frame(con: duckdb.DuckDBPyConnection, layer: MapLayer) -> pl.DataFrame:
    return run_layer(con, layer.sql)[0]


def ranked(rows: pl.DataFrame, areas: Areas, value: str) -> pl.DataFrame:
    """A layer of areas as a ranking: each area's name and its number, largest first. A
    layer may name its areas by key; a ranking is read by name."""
    if areas.name in rows.columns:
        named = rows
    else:
        names = pl.DataFrame(
            [(f["properties"]["id"], f["properties"]["name"]) for f in areas.shapes["features"]],
            schema=[areas.id, areas.name],
            orient="row",
        )
        named = rows.join(names, on=areas.id, how="left")
    return named.select(areas.name, value).drop_nulls(value).sort(value, descending=True)
