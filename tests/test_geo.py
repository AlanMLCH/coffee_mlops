"""The geospatial step is where a bug looks like data rather than a crash.

A wrong projection, a swapped axis order or a mis-decoded attribute table all produce a
table that loads, joins and charts perfectly while placing coffee shops in the wrong
place. These tests pin the three of them down against a recorded boundary layer.

The fixture is shaped exactly like INEGI's download - a shapefile under
`conjunto_de_datos/` inside a ZIP, Latin-1 attributes, geometry in the layer's own
projection, the real CVEGEO and NOMGEO values - but its shapes are a 4x4 grid over the
city, so which borough a point lands in is arbitrary, fixed, and easy to assert.
"""

from pathlib import Path

import duckdb
import polars as pl
import pytest

from mlops_core.config import SpatialConfig
from mlops_core.data import geo
from mlops_core.data.geo import AREA_COLUMNS, attribute_points, read_areas, spatial_connection

ARCHIVE = Path(__file__).parent / "fixtures" / "cdmx_boroughs_sample.zip"
MEMBER = "conjunto_de_datos/09mun.shp"
LAYER = SpatialConfig(
    encoding="ISO-8859-1",
    crs="EPSG:6372",
    id_column="CVEGEO",
    name_column="NOMGEO",
    expected_features=16,
)


@pytest.fixture
def areas() -> pl.DataFrame:
    return read_areas(ARCHIVE, MEMBER, LAYER)


def test_a_layer_is_read_in_place_and_comes_out_decoded(areas: pl.DataFrame) -> None:
    assert list(areas.columns) == AREA_COLUMNS
    assert areas.height == 16
    # Accented names come out as names, not as mojibake: the attribute table is
    # Latin-1 and nothing in the file says so, so the encoding is configuration.
    assert "Cuauhtémoc" in areas["area_name"].to_list()
    assert areas["area_id"].to_list() == sorted(areas["area_id"].to_list())
    assert areas["area_km2"].min() > 0


def test_the_feature_count_is_a_contract() -> None:
    with pytest.raises(ValueError, match="16 features, expected 17"):
        read_areas(ARCHIVE, MEMBER, LAYER.model_copy(update={"expected_features": 17}))


def test_points_land_in_the_area_that_contains_them(areas: pl.DataFrame) -> None:
    points = pl.DataFrame(
        {
            "shop_id": ["north", "south", "at sea"],
            "latitude": [19.45, 19.3048187, 0.0],
            "longitude": [-99.15, -99.1022689, 0.0],
        }
    )

    placed = attribute_points(points, areas, "latitude", "longitude")

    assert dict(zip(placed["shop_id"], placed["area_id"], strict=True)) == {
        "north": "09016",
        "south": "09008",
        # Outside every polygon: kept, with nothing claimed about where it is.
        "at sea": None,
    }


def test_the_wrong_axis_order_would_not_have_been_caught_by_types(areas: pl.DataFrame) -> None:
    """Longitude first. The same numbers the other way round are a valid point in China,
    which is exactly why this is asserted and not assumed."""
    swapped = pl.DataFrame({"shop_id": ["x"], "latitude": [-99.15], "longitude": [19.45]})

    placed = attribute_points(swapped, areas, "latitude", "longitude")

    assert placed["area_id"].to_list() == [None]


def test_no_points_is_not_an_error(areas: pl.DataFrame) -> None:
    """A domain whose register came back empty must still build its table."""
    empty = pl.DataFrame(
        schema={"shop_id": pl.String, "latitude": pl.Float64, "longitude": pl.Float64}
    )

    placed = attribute_points(empty, areas, "latitude", "longitude")

    assert placed.is_empty()
    assert {"area_id", "area_name"} <= set(placed.columns)


def test_overlapping_areas_are_refused_rather_than_counted_twice(areas: pl.DataFrame) -> None:
    """Two areas over one point duplicate its row, and every count downstream inflates."""
    one = areas.filter(pl.col("area_id") == "09016")
    overlapping = pl.concat([one, one.with_columns(pl.lit("09999").alias("area_id"))])
    points = pl.DataFrame({"shop_id": ["x"], "latitude": [19.45], "longitude": [-99.15]})

    with pytest.raises(ValueError, match="the areas overlap"):
        attribute_points(points, overlapping, "latitude", "longitude")


def test_a_missing_extension_says_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extension downloads on first use, so the failure mode is an offline machine."""

    class Offline:
        def install_extension(self, name: str) -> None:
            raise duckdb.Error("could not download")

        def close(self) -> None:
            pass

    monkeypatch.setattr(geo.duckdb, "connect", Offline)

    with pytest.raises(RuntimeError, match="spatial"):
        spatial_connection()
