"""What a query's result can be drawn as, and the chart itself, as Vega-Lite.

The agent answers with a table; this module decides how to show it. A chart is small and
checkable - a kind, and which column goes where - so it can be chosen three ways: by
rules from the result's shape (a date and a number are a line, a place and a number a
map), by a person in the app, or by an MCP client's model. Whoever chooses, the choice
is checked against the columns before anything is drawn, the way the agent's answers
are checked against their evidence. Rules, not a model, choose by default: the query
already shaped the table for the question, so its shape is most of the choice, and a
rule is right the same way every time.

Drawn as Vega-Lite: the app renders the spec live, and MCP clients get the same spec as
a PNG (`png`), so the two never disagree about a chart.
"""

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from mlops_core.explore.style import SEQUENTIAL

MAX_SERIES = 12  # values a colour can tell apart; with more, the colour says nothing
LATITUDE = ("latitude", "lat")
LONGITUDE = ("longitude", "lon", "lng")
VEGA_LITE = "https://vega.github.io/schema/vega-lite/v5.json"
HEIGHT = 340
MAP_HEIGHT = 460  # a city is taller than a bar chart
LONG_LABEL = 10  # characters a name under an upright bar can have before it is cut
MANY_BARS = 12  # upright bars side by side before their names start to be skipped
BAR_STEP = 24  # pixels per lying bar: room for its name, which is never skipped
LABEL_PIXELS = 220  # how much of a lying bar's name shows before it is cut

ChartKind = Literal["bar", "line", "scatter", "heatmap", "points", "areas", "table"]


class Chart(BaseModel):
    """How to draw a query's result: the kind, and which of its columns goes where."""

    kind: ChartKind = Field(
        description="bar: a number per category; line: a number over time; scatter: two "
        "numbers; heatmap: a number for each pair of two categories (x across, y down, "
        "color the number); points: rows with latitude and longitude, on a map; areas: a "
        "number per area, as a map; table: the rows as they are"
    )
    x: str | None = Field(
        default=None,
        description="bar: the category column; line: the time column; scatter: a number; "
        "heatmap: the category across",
    )
    y: str | None = Field(
        default=None,
        description="The number drawn: bar height, line, scatter's vertical axis, colour; "
        "a heatmap's category down",
    )
    color: str | None = Field(
        default=None,
        description=f"A column whose values split the marks into series, {MAX_SERIES} at "
        "most; a heatmap's number, drawn as each cell's shade",
    )
    title: str = ""


@dataclass(frozen=True)
class Areas:
    """The places a result can name, by key or by name, and their outlines."""

    id: str  # the column that holds an area's key in the domain's tables
    name: str  # and its name
    shapes: dict[str, Any]  # a GeoJSON FeatureCollection; properties `id` and `name`


def frame(columns: list[str], rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """A query's rows as a frame, typed from every value, not the first hundred."""
    return pl.DataFrame(rows, schema=columns, orient="row", infer_schema_length=None)


def infer_chart(result: pl.DataFrame, areas: Areas | None = None, title: str = "") -> Chart:
    """The chart a result's shape asks for: a map for places, a line for time, bars for
    categories, a scatter for two numbers - or the table, when there is nothing to draw."""
    latitude, longitude = _coordinates(result)
    if latitude and longitude and not result.is_empty():
        return Chart(kind="points", color=_series(result, _labels(result)), title=title)
    periods = _periods(result)
    numbers = _measures(result, [c for c in _numbers(result) if c not in periods])
    area = _area_column(result, areas)
    if area and numbers:
        return Chart(kind="areas", y=numbers[0], title=title)
    if result.height < 2 or not numbers:
        return Chart(kind="table", title=title)
    labels = _labels(result)
    if periods:
        return Chart(kind="line", x=periods[0], y=numbers[0], color=_series(result, labels),
                     title=title)  # fmt: skip
    if labels:
        return Chart(kind="bar", x=labels[0], y=numbers[0], color=_series(result, labels[1:]),
                     title=title)  # fmt: skip
    if len(numbers) >= 2:
        return Chart(kind="scatter", x=numbers[0], y=numbers[1], title=title)
    return Chart(kind="table", title=title)


def check_chart(chart: Chart, result: pl.DataFrame, areas: Areas | None = None) -> list[str]:
    """What stops the chart from being drawn from this result; nothing when it can be."""
    missing = [
        f"{role}: the result has no column {name!r} (it has {', '.join(result.columns)})"
        for role, name in (("x", chart.x), ("y", chart.y), ("color", chart.color))
        if name is not None and name not in result.columns
    ]
    if missing:
        return missing
    problems = []
    needs = {"bar": ("x", "y"), "line": ("x", "y"), "scatter": ("x", "y"), "areas": ("y",),
             "heatmap": ("x", "y", "color")}  # fmt: skip
    problems += [f"a {chart.kind} chart needs {role}" for role in needs.get(chart.kind, ())
                 if getattr(chart, role) is None]  # fmt: skip
    numeric = set(_numbers(result))
    if chart.kind in ("bar", "line", "scatter", "areas") and chart.y and chart.y not in numeric:
        problems.append(f"y must be a number; {chart.y!r} is not")
    if chart.kind == "scatter" and chart.x and chart.x not in numeric:
        problems.append(f"a scatter's x must be a number; {chart.x!r} is not")
    if chart.kind == "points" and None in _coordinates(result):
        problems.append("points need a latitude and a longitude column")
    if chart.kind == "areas" and _area_column(result, areas) is None:
        known = f"{areas.id!r} or {areas.name!r}" if areas else "an area table"
        problems.append(f"areas need a column naming the area: {known}")
    if chart.kind == "heatmap":  # its colour is a number's shade, not a few series
        if chart.color and chart.color not in numeric:
            problems.append(f"a heatmap's color must be a number; {chart.color!r} is not")
        return problems
    if chart.color and result[chart.color].n_unique() > MAX_SERIES:
        problems.append(f"{chart.color!r} has more than {MAX_SERIES} values to colour by")
    return problems


def vega_lite(
    chart: Chart, result: pl.DataFrame, areas: Areas | None = None
) -> dict[str, Any] | None:
    """The chart as a Vega-Lite spec with its data inline; None for a table. Check it
    first: this draws what it is told."""
    if chart.kind == "table":
        return None
    records = _records(result)
    tooltip = [{"field": c, "type": _type(result, c)} for c in result.columns]
    spec: dict[str, Any] = {"$schema": VEGA_LITE, "width": "container", "height": HEIGHT}
    if chart.title:
        spec["title"] = chart.title
    color: dict[str, Any] = (
        {"color": {"field": chart.color, "type": "nominal"}} if chart.color is not None else {}
    )
    if chart.kind == "points":
        latitude, longitude = _coordinates(result)
        points = {
            "data": {"values": records},
            "mark": {"type": "circle", "size": 22, "opacity": 0.75},
            "encoding": {
                "latitude": {"field": latitude, "type": "quantitative"},
                "longitude": {"field": longitude, "type": "quantitative"},
                "tooltip": tooltip,
                **color,
            },
        }
        outline = [_outline(areas)] if areas else []
        return spec | {
            "height": MAP_HEIGHT,
            "projection": {"type": "mercator"},
            "layer": [*outline, points],
        }
    if chart.kind == "areas":
        assert areas is not None and chart.y is not None  # check_chart's to say
        area = _area_column(result, areas)
        key = "properties.id" if area == areas.id else "properties.name"
        return spec | {
            "height": MAP_HEIGHT,
            "data": {"values": areas.shapes, "format": {"type": "json", "property": "features"}},
            "transform": [
                {
                    "lookup": key,
                    "from": {"data": {"values": records}, "key": area, "fields": [chart.y]},
                }
            ],
            "projection": {"type": "mercator"},
            "mark": {"type": "geoshape", "stroke": "white", "strokeWidth": 0.6},
            "encoding": {
                "color": {
                    "field": chart.y,
                    "type": "quantitative",
                    "scale": {"range": SEQUENTIAL},
                    "title": _title(chart.y),
                },
                "tooltip": [
                    {"field": "properties.name", "type": "nominal", "title": "area"},
                    {"field": chart.y, "type": "quantitative"},
                ],
            },
        }
    if chart.kind == "heatmap":
        # Two categories in their own order - hours, weekdays - and the number as a shade:
        # a grid reads where seven lines over the same hours tangle.
        return spec | {
            "data": {"values": records},
            "mark": {"type": "rect"},
            "encoding": {
                "x": {"field": chart.x, "type": "ordinal", "title": _title(chart.x)},
                "y": {"field": chart.y, "type": "ordinal", "title": _title(chart.y)},
                "color": {
                    "field": chart.color,
                    "type": "quantitative",
                    "scale": {"range": SEQUENTIAL},
                    "title": _title(chart.color),
                },
                "tooltip": tooltip,
            },
        }
    if color:
        color["color"]["title"] = _title(chart.color)
    if chart.kind == "bar" and chart.x is not None and _long_labels(result, chart.x):
        # Names too long to stand under a bar: the bars lie down, one row per name, and
        # the chart grows with them instead of skipping and cutting labels.
        assert chart.y is not None  # check_chart's to say
        series = result[chart.color].n_unique() if chart.color else 1
        return spec | {
            "height": {"step": BAR_STEP * series},
            "data": {"values": records},
            "mark": {"type": "bar"},
            "encoding": {
                "y": {
                    "field": chart.x,
                    "type": "nominal",
                    "sort": "-x",
                    "title": None,
                    "axis": {"labelOverlap": False, "labelLimit": LABEL_PIXELS},
                },
                "x": {"field": chart.y, "type": "quantitative", "title": _title(chart.y)},
                "tooltip": tooltip,
                **color,
                **({"yOffset": {"field": chart.color}} if chart.color else {}),
            },
        }
    x_type = "quantitative" if chart.kind == "scatter" else _type(result, chart.x)
    if chart.kind == "bar":
        x_type = "nominal"
    # Bars of numbers (a bag's grams) stand in their own order; bars of names, by height.
    order = "ascending" if chart.x is not None and result[chart.x].dtype.is_numeric() else "-y"
    encoding: dict[str, Any] = {
        "x": {"field": chart.x, "type": x_type, "title": _title(chart.x)}
        | ({"sort": order, "axis": {"labelAngle": -35}} if chart.kind == "bar" else {}),
        "y": {"field": chart.y, "type": "quantitative", "title": _title(chart.y)}
        # A line's change is the story; from zero, a 15% rise is a flat line.
        | ({"scale": {"zero": False}} if chart.kind in ("line", "scatter") else {}),
        "tooltip": tooltip,
        **color,
    }
    if chart.kind == "bar" and chart.color is not None:
        encoding["xOffset"] = {"field": chart.color}  # side by side: segments are compared
    mark = {"bar": {"type": "bar"}, "line": {"type": "line", "point": True},
            "scatter": {"type": "circle", "size": 50}}[chart.kind]  # fmt: skip
    return spec | {"data": {"values": records}, "mark": mark, "encoding": encoding}


def png(spec: dict[str, Any], width: int = 720) -> bytes:
    """The spec as a PNG, drawn by Vega itself (vl-convert): what an MCP client is shown.
    A width left to the page ("container") is fixed here, since there is no page."""
    import vl_convert  # the mcp extra's; imported where used

    fixed = spec | {"width": width} if spec.get("width") == "container" else spec
    image: bytes = vl_convert.vegalite_to_png(fixed, scale=2)
    return image


def _title(column: str | None) -> str | None:
    """An axis says "price per kg", not "price_per_kg"."""
    return column.replace("_", " ") if column else None


def _long_labels(result: pl.DataFrame, column: str) -> bool:
    """Whether a category's names are too long to stand under a bar: longer than
    LONG_LABEL characters, or too many to stand side by side."""
    names = [str(value) for value in result[column].unique().to_list()]
    return max((len(name) for name in names), default=0) > LONG_LABEL or len(names) > MANY_BARS


def _outline(areas: Areas) -> dict[str, Any]:
    return {
        "data": {"values": areas.shapes, "format": {"type": "json", "property": "features"}},
        "mark": {"type": "geoshape", "filled": False, "stroke": "#9a9890", "strokeWidth": 0.6},
    }


def _coordinates(result: pl.DataFrame) -> tuple[str | None, str | None]:
    columns = {c.lower(): c for c in result.columns}
    latitude = next((columns[n] for n in LATITUDE if n in columns), None)
    longitude = next((columns[n] for n in LONGITUDE if n in columns), None)
    return latitude, longitude


def _numbers(result: pl.DataFrame) -> list[str]:
    coordinates = set(_coordinates(result))
    return [c for c in result.columns if result[c].dtype.is_numeric() and c not in coordinates]


def _measures(result: pl.DataFrame, numbers: list[str]) -> list[str]:
    """The numbers, the one the rows are sorted by first: a query orders by what the
    question asked about (`ORDER BY shops_per_km2 DESC`), and a count beside it is context."""
    if result.height < 3:
        return numbers
    ordered = [c for c in numbers if result[c].is_sorted() or result[c].is_sorted(descending=True)]
    return ordered + [c for c in numbers if c not in ordered]


def is_period(result: pl.DataFrame, column: str) -> bool:
    """Whether a column is time: dates, and whole numbers named as years (`year`,
    `market_year`). Time is drawn as a line and never cut to its largest values."""
    dtype = result[column].dtype
    return dtype.is_temporal() or (
        dtype.is_integer() and (column == "year" or column.endswith("_year"))
    )


def _periods(result: pl.DataFrame) -> list[str]:
    return [c for c in result.columns if is_period(result, c)]


def _labels(result: pl.DataFrame) -> list[str]:
    return [c for c in result.columns if result[c].dtype in (pl.String, pl.Boolean, pl.Categorical)]


def _series(result: pl.DataFrame, candidates: list[str]) -> str | None:
    """The first column that splits the rows into a few series, if any does."""
    return next((c for c in candidates if 1 < result[c].n_unique() <= MAX_SERIES), None)


def _area_column(result: pl.DataFrame, areas: Areas | None) -> str | None:
    if areas is None:
        return None
    return next((c for c in result.columns if c in (areas.id, areas.name)), None)


def _type(result: pl.DataFrame, column: str | None) -> str:
    dtype = result[column].dtype if column else pl.String
    if dtype.is_temporal():
        return "temporal"
    if column is not None and is_period(result, column):
        return "ordinal"
    return "quantitative" if dtype.is_numeric() else "nominal"


def _records(result: pl.DataFrame) -> list[dict[str, Any]]:
    """Rows as JSON can carry them: dates as ISO text, decimals as floats."""
    return [{k: _plain(v) for k, v in row.items()} for row in result.iter_rows(named=True)]


def _plain(value: object) -> object:
    if isinstance(value, dt.date | dt.datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value
