"""Charts from query results: the rules that choose one, the checks that refuse one, and
the spec that both the app and MCP clients draw - plus the outlines a map needs, read
out of WKB without a geometry library."""

import struct
from contextlib import closing
from datetime import date
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from mlops_core.config import ExploreConfig
from mlops_core.data.geo import spatial_connection
from mlops_core.explore.charts import (
    MAX_SERIES,
    Areas,
    Chart,
    check_chart,
    frame,
    infer_chart,
    png,
    vega_lite,
)
from mlops_core.explore.shapes import feature_collection, geometry


def wkb_of(wkt: str) -> bytes:
    """WKB as DuckDB's spatial extension writes it: what the area table holds."""
    with closing(spatial_connection()) as con:
        (wkb,) = con.sql(f"SELECT ST_AsWKB(ST_GeomFromText('{wkt}'))").fetchone()  # type: ignore[misc]
    return bytes(wkb)


NORTH = wkb_of("POLYGON((0 1, 1 1, 1 2, 0 2, 0 1))")
SOUTH = wkb_of("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
AREAS = Areas(
    "area_id", "area", feature_collection([("a1", "North", NORTH), ("a2", "South", SOUTH)])
)


# --- Outlines --------------------------------------------------------------------------------


def test_a_polygon_with_a_hole_and_a_multipolygon_come_out_as_geojson() -> None:
    holed = geometry(wkb_of("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 1))"))
    parts = geometry(wkb_of("MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((5 5, 6 5, 6 6, 5 5)))"))

    assert holed["type"] == "Polygon" and len(holed["coordinates"]) == 2
    assert holed["coordinates"][1][0] == [1.0, 1.0]
    assert parts == {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
            [[[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 5.0]]],
        ],
    }


def test_coordinates_are_rounded_to_a_metre_or_so() -> None:
    shape = geometry(
        wkb_of("POLYGON((-99.123456789 19.1, -99.1 19.1, -99.1 19.2, -99.123456789 19.1))")
    )

    assert shape["coordinates"][0][0] == [-99.12346, 19.1]


def test_big_endian_wkb_is_read_too() -> None:
    ring = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    wkb = struct.pack(">BIII", 0, 3, 1, len(ring)) + b"".join(struct.pack(">dd", *p) for p in ring)

    assert geometry(wkb)["coordinates"] == [[list(p) for p in ring]]


def test_anything_but_an_area_is_refused() -> None:
    with pytest.raises(ValueError, match="WKB type 1 is not an area"):
        geometry(wkb_of("POINT(1 2)"))
    line = struct.pack("<BIII", 1, 2, 2, 0)  # a multipolygon holding a linestring
    with pytest.raises(ValueError, match="holds WKB type 2"):
        geometry(struct.pack("<BII", 1, 6, 1) + line)


def test_areas_become_features_with_their_key_and_name() -> None:
    features = AREAS.shapes["features"]

    assert [f["properties"] for f in features] == [
        {"id": "a1", "name": "North"},
        {"id": "a2", "name": "South"},
    ]


# --- Which chart a result's shape asks for -----------------------------------------------------


def shops() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": [f"shop {i}" for i in range(20)],
            "source": ["denue", "osm"] * 10,
            "latitude": [19.4 + i / 1000 for i in range(20)],
            "longitude": [-99.1] * 20,
        }
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        # Places: on the map, coloured by the column with a few values, not by name.
        (shops(), Chart(kind="points", color="source")),
        # A number per area, by key or by name.
        (pl.DataFrame({"area_id": ["a1", "a2"], "shops": [3, 5]}), Chart(kind="areas", y="shops")),
        (pl.DataFrame({"area": ["North"], "price": [380.0]}), Chart(kind="areas", y="price")),
        # Time: a date, or a whole number named as a year; split by the label beside it.
        (
            pl.DataFrame({"fortnight": [date(2026, 7, 1), date(2026, 7, 16)] * 2,
                          "product": ["ground", "ground", "instant", "instant"],
                          "price": [380.0, 395.0, 900.0, 958.3]}),
            Chart(kind="line", x="fortnight", y="price", color="product"),
        ),
        (
            pl.DataFrame({"market_year": [2024, 2025], "production": [4.1, 4.3]}),
            Chart(kind="line", x="market_year", y="production"),
        ),
        # Categories: bars, split by a second label when it has few values.
        (
            pl.DataFrame({"chain": ["A", "A", "B"], "product": ["ground", "instant", "ground"],
                          "price": [1.0, 2.0, 3.0]}),
            Chart(kind="bar", x="chain", y="price", color="product"),
        ),
        (pl.DataFrame({"state": ["X", "Y"], "price": [1.0, 2.0]}),
         Chart(kind="bar", x="state", y="price")),
        # Two numbers: a scatter.
        (pl.DataFrame({"altitude": [1.0, 2.0, 3.0], "points": [80.0, 82.0, 85.0]}),
         Chart(kind="scatter", x="altitude", y="points")),
        # Nothing to draw: one row, no number, a single number, no rows.
        (pl.DataFrame({"state": ["X"], "price": [1.0]}), Chart(kind="table")),
        (pl.DataFrame({"state": ["X", "Y"]}), Chart(kind="table")),
        (pl.DataFrame({"price": [1.0, 2.0]}), Chart(kind="table")),
        (shops().clear(), Chart(kind="table")),
    ],
)  # fmt: skip
def test_the_result_s_shape_chooses_the_chart(result: pl.DataFrame, expected: Chart) -> None:
    assert infer_chart(result, AREAS) == expected


def test_without_areas_a_named_place_is_just_a_category() -> None:
    result = pl.DataFrame({"area": ["North", "South"], "price": [1.0, 2.0]})

    assert infer_chart(result, None, title="t") == Chart(kind="bar", x="area", y="price", title="t")


def test_a_query_s_rows_are_typed_from_every_value() -> None:
    result = frame(["month", "price"], [(date(2026, 7, 1), None)] * 120 + [(date(2026, 8, 1), 3)])

    assert result.schema == {"month": pl.Date, "price": pl.Int64}


# --- What stops a chart --------------------------------------------------------------------


PRICES = pl.DataFrame(
    {"area_id": ["a1", "a2"], "chain": ["A", "B"], "price": [1.0, 2.0], "name": ["x", "y"]}
)


@pytest.mark.parametrize(
    ("chart", "problem"),
    [
        (Chart(kind="bar", x="store", y="price"), "no column 'store'"),
        (Chart(kind="bar", x="chain"), "a bar chart needs y"),
        (Chart(kind="bar", x="price", y="chain"), "y must be a number; 'chain' is not"),
        (Chart(kind="scatter", x="chain", y="price"), "a scatter's x must be a number"),
        (Chart(kind="points"), "points need a latitude and a longitude column"),
    ],
)
def test_a_chart_the_result_cannot_carry_is_refused(chart: Chart, problem: str) -> None:
    assert any(problem in found for found in check_chart(chart, PRICES, AREAS))


def test_areas_need_a_column_that_names_one() -> None:
    by_chain = PRICES.select("chain", "price")

    assert check_chart(Chart(kind="areas", y="price"), by_chain, AREAS) == [
        "areas need a column naming the area: 'area_id' or 'area'"
    ]
    assert "an area table" in check_chart(Chart(kind="areas", y="price"), PRICES, None)[0]


def test_a_colour_with_too_many_values_says_nothing() -> None:
    many = pl.DataFrame({"shop": [str(i) for i in range(MAX_SERIES + 1)], "price": [1.0] * 13})

    (problem,) = check_chart(Chart(kind="bar", x="shop", y="price", color="shop"), many)

    assert "more than 12 values" in problem


def test_a_table_and_the_inferred_charts_pass_their_checks() -> None:
    assert check_chart(Chart(kind="table"), PRICES) == []
    assert check_chart(Chart(kind="areas", y="price"), PRICES, AREAS) == []
    assert check_chart(infer_chart(shops(), AREAS), shops(), AREAS) == []


# --- The spec ----------------------------------------------------------------------------------


def test_bars_are_sorted_and_segments_stand_side_by_side() -> None:
    result = pl.DataFrame(
        {"chain": ["A", "B"], "product": ["g", "i"], "price": [Decimal("1.5"), Decimal("2")]}
    )

    spec = vega_lite(
        Chart(kind="bar", x="chain", y="price", color="product", title="By chain"), result
    )

    assert spec is not None
    assert spec["title"] == "By chain" and spec["mark"] == {"type": "bar"}
    assert spec["encoding"]["x"]["sort"] == "-y" and spec["encoding"]["xOffset"] == {
        "field": "product"
    }
    assert spec["data"]["values"][0]["price"] == 1.5  # a decimal, as JSON carries it


def test_a_line_over_dates_is_temporal_and_over_years_ordinal() -> None:
    dated = pl.DataFrame({"month": [date(2026, 7, 1), date(2026, 8, 1)], "price": [1.0, 2.0]})
    yearly = pl.DataFrame({"year": [2024, 2025], "price": [1.0, 2.0]})

    by_date = vega_lite(Chart(kind="line", x="month", y="price"), dated)
    by_year = vega_lite(Chart(kind="line", x="year", y="price"), yearly)

    assert by_date is not None and by_year is not None
    assert by_date["encoding"]["x"]["type"] == "temporal"
    assert by_date["data"]["values"][0]["month"] == "2026-07-01"
    assert by_year["encoding"]["x"]["type"] == "ordinal"
    assert by_date["mark"] == {"type": "line", "point": True} and "title" not in by_date


def test_a_scatter_puts_a_number_on_each_axis() -> None:
    result = pl.DataFrame({"altitude": [1.0, 2.0], "points": [80.0, 82.0]})

    spec = vega_lite(Chart(kind="scatter", x="altitude", y="points"), result)

    assert spec is not None and spec["encoding"]["x"]["type"] == "quantitative"


def test_points_sit_on_the_areas_outlines() -> None:
    with_outline = vega_lite(Chart(kind="points", color="source"), shops(), AREAS)
    alone = vega_lite(Chart(kind="points"), shops())

    assert with_outline is not None and alone is not None
    assert [layer["mark"]["type"] for layer in with_outline["layer"]] == ["geoshape", "circle"]
    assert with_outline["layer"][1]["encoding"]["color"] == {"field": "source", "type": "nominal"}
    assert len(alone["layer"]) == 1 and alone["projection"] == {"type": "mercator"}


def test_areas_are_coloured_by_looking_up_the_result_by_key_or_by_name() -> None:
    by_key = vega_lite(Chart(kind="areas", y="price"), PRICES.select("area_id", "price"), AREAS)
    by_name = vega_lite(
        Chart(kind="areas", y="price"), pl.DataFrame({"area": ["North"], "price": [3.0]}), AREAS
    )

    assert by_key is not None and by_name is not None
    assert by_key["transform"][0]["lookup"] == "properties.id"
    assert by_key["transform"][0]["from"]["key"] == "area_id"
    assert by_name["transform"][0]["lookup"] == "properties.name"
    assert by_key["data"]["values"] is AREAS.shapes


def test_a_table_is_not_drawn() -> None:
    assert vega_lite(Chart(kind="table"), PRICES) is None


def test_the_same_spec_is_drawn_as_a_png_for_a_client_without_a_page(tmp_path: Path) -> None:
    spec = vega_lite(Chart(kind="bar", x="chain", y="price"), PRICES)
    assert spec is not None

    image = png(spec)
    fixed = png(spec | {"width": 200})

    assert image.startswith(b"\x89PNG") and fixed.startswith(b"\x89PNG")
    assert spec["width"] == "container"  # the page's spec is left as it was


# --- The app's config --------------------------------------------------------------------------


def test_a_layer_of_areas_needs_the_areas_it_draws() -> None:
    view = {"latitude": 19.4, "longitude": -99.1, "zoom": 10}
    layer = {"name": "density", "kind": "areas", "sql": "SELECT 1"}

    with pytest.raises(ValidationError, match=r"draw areas, but `explore\.areas` names none"):
        ExploreConfig(title="t", view=view, layers=[layer])  # type: ignore[arg-type]


# --- The map's data, through the agent's locked session -------------------------------------


def test_areas_and_layers_are_read_as_any_question_s_sql_is(tmp_path: Path) -> None:
    from mlops_core.agent.sql import Refused, read_only
    from mlops_core.config import AreasConfig, MapLayer
    from mlops_core.explore.layers import layer_frame, load_areas, run_layer
    from mlops_core.storage import write_table

    outlines = {"area_id": ["a1", "a2"], "area": ["North", "South"], "boundary": [NORTH, SOUTH]}
    write_table(pl.DataFrame(outlines), tmp_path / "clean" / "areas", inputs={})
    places = shops().head(3)
    write_table(places, tmp_path / "clean" / "shops", inputs={})
    con = read_only(tmp_path)

    areas = load_areas(con, AreasConfig(table="clean.areas", id="area_id", name="area",
                                        boundary="boundary"))  # fmt: skip
    first, cut = run_layer(con, "SELECT * FROM clean.shops", max_rows=2)
    layer = layer_frame(con, MapLayer(name="shops", kind="points", sql="SELECT * FROM clean.shops"))

    assert [f["properties"]["name"] for f in areas.shapes["features"]] == ["North", "South"]
    assert (first.height, cut) == (2, True)
    assert layer.columns == places.columns and layer.height == 3
    with pytest.raises(Refused):
        run_layer(con, "DROP VIEW clean.shops")


def test_a_map_of_areas_waits_for_its_table(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from mlops_core.agent.sql import read_only
    from mlops_core.explore.layers import areas_if_built
    from mlops_core.storage import write_table

    write_table(shops(), tmp_path / "clean" / "shops", inputs={})
    con = read_only(tmp_path)
    view = {"latitude": 19.4, "longitude": -99.1, "zoom": 10}
    unbuilt = ExploreConfig(
        title="t", view=view,  # type: ignore[arg-type]
        areas={"table": "clean.areas", "id": "a", "name": "b", "boundary": "c"},  # type: ignore[arg-type]
    )  # fmt: skip

    assert areas_if_built(con, None) is None
    assert areas_if_built(con, ExploreConfig(title="t", view=view)) is None  # type: ignore[arg-type]
    assert areas_if_built(con, unbuilt) is None
    assert "clean.areas is not built yet" in caplog.text
