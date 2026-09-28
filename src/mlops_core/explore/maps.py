"""The explorer's map as deck.gl layers (pydeck): a number per area, raised as columns on
a tilted map and coloured from light to dark; places as points coloured by a category;
and places counted in hexagons, where thousands of overlapping dots would hide how many
there are. The YAML's layers are drawn this way before any question, and an answer about
places is drawn the same way after one.

Every object carries a `tooltip` line, since a map shows several layers under one tooltip
template - which is also why the hexagons are counted here rather than by deck.gl's own
HexagonLayer, whose objects could not carry one.
"""

import math
from collections import Counter
from typing import Any

import polars as pl
import pydeck as pdk

from mlops_core.config import MapView
from mlops_core.explore.charts import LATITUDE, LONGITUDE, Areas
from mlops_core.explore.style import SEQUENTIAL, SERIES, Rgb, rgb

PALETTE = [rgb(color) for color in SERIES]  # a colour legend never needs more
STOPS = [rgb(color) for color in SEQUENTIAL]  # a number, from little to much
NO_VALUE = (205, 199, 190)  # an area the layer says nothing about
MAX_ELEVATION = 2_500  # metres the area with the largest number rises to
HEXAGON_METRES = 400  # centre to corner: a few blocks of a city
METRES_PER_DEGREE = 111_320  # of latitude; of longitude, times the cosine of it


def area_layer(
    areas: Areas, values: pl.DataFrame, value: str, name: str, raised: bool = True
) -> pdk.Layer:
    """Each area coloured and raised by its number in `values`, found by the area's key or
    name; an area without one is flat and grey. Not `raised`, the areas lie flat: places
    drawn over them would otherwise stand inside their columns."""
    key = next(c for c in values.columns if c in (areas.id, areas.name))
    field = "id" if key == areas.id else "name"
    numbers = {row[key]: row[value] for row in values.iter_rows(named=True)}
    top = max((v for v in numbers.values() if v is not None), default=0) or 1
    features = []
    for feature in areas.shapes["features"]:
        number = numbers.get(feature["properties"][field])
        share = float(number) / float(top) if number is not None else 0.0
        properties = feature["properties"] | {
            "tooltip": f"{feature['properties']['name']}: {_label(number)} ({value})",
            "fill": list(ramp(share)) if number is not None else list(NO_VALUE),
            "elevation": share * MAX_ELEVATION,
        }
        features.append(feature | {"properties": properties})
    return pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": features},
        id=name,
        extruded=raised,
        wireframe=True,
        opacity=0.75,
        get_fill_color="properties.fill",
        get_elevation="properties.elevation",
        get_line_color=[255, 255, 255],
        pickable=True,
        auto_highlight=True,
    )


def point_layer(
    places: pl.DataFrame, color: str | None, name: str
) -> tuple[pdk.Layer, dict[str, Rgb]]:
    """Each row a point at its latitude and longitude, coloured by `color`'s value; and the
    colour each value got, for a legend."""
    latitude, longitude = coordinates(places)
    values = sorted({str(v) for v in places[color].to_list()}) if color else []
    legend = {value: PALETTE[i % len(PALETTE)] for i, value in enumerate(values)}
    described = [c for c in places.columns if c not in (latitude, longitude)]
    records = [
        {
            "position": [row[longitude], row[latitude]],
            "color": list(legend[str(row[color])] if color else PALETTE[0]),
            "tooltip": " · ".join(f"{c}: {_label(row[c])}" for c in described),
        }
        for row in places.iter_rows(named=True)
        if row[latitude] is not None and row[longitude] is not None
    ]
    layer = pdk.Layer(
        "ScatterplotLayer",
        records,
        id=name,
        get_position="position",
        get_fill_color="color",
        get_radius=45,
        radius_min_pixels=2,
        radius_max_pixels=9,
        opacity=0.85,
        pickable=True,
        auto_highlight=True,
    )
    return layer, legend


def hexagons(places: pl.DataFrame, radius: float = HEXAGON_METRES) -> list[dict[str, Any]]:
    """How many places fall in each hexagon of a grid `radius` metres from centre to
    corner: the hexagon's centre, as longitude and latitude, and its count.

    The grid is laid on a flat projection around the places' mean latitude - at a city's
    scale the earth is flat to a few metres - with pointy-top hexagons in axial
    coordinates, each point rounded to its nearest centre in cube coordinates (Red Blob
    Games' "Hexagonal Grids" gives the arithmetic).
    """
    latitude, longitude = coordinates(places)
    points = places.select(latitude, longitude).drop_nulls().rows()
    if not points:
        return []
    middle = sum(lat for lat, _ in points) / len(points)
    across = METRES_PER_DEGREE * math.cos(math.radians(middle))  # metres per degree of longitude
    counts = Counter(
        _nearest_hexagon(lon * across / radius, lat * METRES_PER_DEGREE / radius)
        for lat, lon in points
    )
    cells = []
    for (q, r), count in counts.items():
        x, y = math.sqrt(3) * (q + r / 2), 1.5 * r  # the centre, in radii
        cells.append(
            {
                "longitude": x * radius / across,
                "latitude": y * radius / METRES_PER_DEGREE,
                "count": count,
            }
        )
    return sorted(cells, key=lambda cell: -cell["count"])


def density_layer(places: pl.DataFrame, name: str, radius: float = HEXAGON_METRES) -> pdk.Layer:
    """Places counted in hexagons, each raised and coloured by its count: where they crowd."""
    cells = hexagons(places, radius)
    top = max((cell["count"] for cell in cells), default=1)
    records = [
        {
            "position": [cell["longitude"], cell["latitude"]],
            "fill": list(ramp(cell["count"] / top)),
            "elevation": cell["count"] / top * MAX_ELEVATION,
            "tooltip": f"{cell['count']:,} {'place' if cell['count'] == 1 else 'places'} "
            f"within about {radius:,.0f} m ({name})",
        }
        for cell in cells
    ]
    return pdk.Layer(
        "ColumnLayer",
        records,
        id=name,
        get_position="position",
        get_fill_color="fill",
        get_elevation="elevation",
        radius=radius * 0.9,  # a thin street between neighbours
        disk_resolution=6,  # a hexagon
        angle=90,  # pointy-top, as the grid is laid
        extruded=True,
        opacity=0.85,
        pickable=True,
        auto_highlight=True,
    )


def deck(view: MapView, layers: list[pdk.Layer]) -> pdk.Deck:
    """The map: the domain's view, tilted, with one tooltip line per object, on Carto's
    light basemap (no key needed): left to Streamlit, a custom theme got the dark one,
    and the ramp's dark end disappeared into it."""
    return pdk.Deck(
        layers=layers,
        map_style=pdk.map_styles.LIGHT,
        initial_view_state=pdk.ViewState(
            latitude=view.latitude, longitude=view.longitude, zoom=view.zoom, pitch=view.pitch
        ),
        tooltip={"text": "{tooltip}"},
    )


def ramp(share: float) -> Rgb:
    """A number's colour: its share of the largest, along the stops from light to dark."""
    share = min(max(share, 0.0), 1.0)
    position = share * (len(STOPS) - 1)
    below = min(int(position), len(STOPS) - 2)
    part = position - below
    red, green, blue = (
        round(low + (high - low) * part)
        for low, high in zip(STOPS[below], STOPS[below + 1], strict=True)
    )
    return red, green, blue


def coordinates(places: pl.DataFrame) -> tuple[str, str]:
    """The latitude and longitude columns, by the names a query gives them."""
    latitude = next(c for c in places.columns if c.lower() in LATITUDE)
    longitude = next(c for c in places.columns if c.lower() in LONGITUDE)
    return latitude, longitude


def _nearest_hexagon(x: float, y: float) -> tuple[int, int]:
    """The axial coordinates of the pointy-top hexagon, of radius 1, whose centre is
    nearest (x, y): the fractional axial coordinates, rounded in cube space."""
    q, r = math.sqrt(3) / 3 * x - y / 3, 2 / 3 * y
    s = -q - r
    rq, rr, rs = round(q), round(r), round(s)
    dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return rq, rr


def _label(value: Any) -> str:
    if value is None:
        return "no data"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}" if isinstance(value, int) else str(value)
