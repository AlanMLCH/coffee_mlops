"""The explorer's map as deck.gl layers (pydeck): a number per area, raised as columns on
a tilted map and coloured from light to deep, and places as points coloured by a
category. The YAML's layers are drawn this way before any question, and an answer about
places is drawn the same way after one.

Every object carries a `tooltip` line, since a map shows several layers under one tooltip
template.
"""

from typing import Any

import polars as pl
import pydeck as pdk

from mlops_core.config import MapView
from mlops_core.explore.charts import LATITUDE, LONGITUDE, Areas

# The house style's categorical hues, then two more: a colour legend never needs more.
PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0),
           (227, 73, 72), (137, 135, 129)]  # fmt: skip
LOW, HIGH = (214, 229, 244), (8, 69, 148)  # the lightest and deepest blue of a number's ramp
NO_VALUE = (200, 200, 196)  # an area the layer says nothing about
MAX_ELEVATION = 2_500  # metres the area with the largest number rises to
Rgb = tuple[int, int, int]


def area_layer(areas: Areas, values: pl.DataFrame, value: str, name: str) -> pdk.Layer:
    """Each area coloured and raised by its number in `values`, found by the area's key or
    name; an area without one is flat and grey."""
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
        extruded=True,
        wireframe=True,
        opacity=0.72,
        get_fill_color="properties.fill",
        get_elevation="properties.elevation",
        get_line_color=[255, 255, 255],
        pickable=True,
    )


def point_layer(
    places: pl.DataFrame, color: str | None, name: str
) -> tuple[pdk.Layer, dict[str, Rgb]]:
    """Each row a point at its latitude and longitude, coloured by `color`'s value; and the
    colour each value got, for a legend."""
    latitude = next(c for c in places.columns if c.lower() in LATITUDE)
    longitude = next(c for c in places.columns if c.lower() in LONGITUDE)
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
        opacity=0.8,
        pickable=True,
    )
    return layer, legend


def deck(view: MapView, layers: list[pdk.Layer]) -> pdk.Deck:
    """The map: the domain's view, tilted, with one tooltip line per object. The map's
    style is left to Streamlit, which follows the page's light or dark theme."""
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=view.latitude, longitude=view.longitude, zoom=view.zoom, pitch=view.pitch
        ),
        tooltip={"text": "{tooltip}"},
    )


def ramp(share: float) -> Rgb:
    """A number's colour: its share of the largest, from light blue to deep."""
    share = min(max(share, 0.0), 1.0)
    red, green, blue = (
        round(low + (high - low) * share) for low, high in zip(LOW, HIGH, strict=True)
    )
    return red, green, blue


def _label(value: Any) -> str:
    if value is None:
        return "no data"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}" if isinstance(value, int) else str(value)
