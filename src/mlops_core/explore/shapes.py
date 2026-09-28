"""Area outlines as GeoJSON, read out of WKB without a geometry library.

A domain's area table keeps each outline as WKB (the core's `geo` writes it so), and both
the app's map and a Vega-Lite chart draw GeoJSON. An administrative area is a polygon or
a multipolygon, and their WKB is a few integers and doubles: reading it here keeps the
explorer off DuckDB's spatial extension, which reading the published layers does not
need - and the agent's locked session could not load anyway.
"""

import struct
from collections.abc import Iterable
from typing import Any

POLYGON, MULTIPOLYGON = 3, 6
DIGITS = 5  # decimal degrees kept: 1 m or so, and a fifth of the payload


def geometry(wkb: bytes, digits: int = DIGITS) -> dict[str, Any]:
    """A WKB polygon or multipolygon as a GeoJSON geometry, coordinates rounded."""
    kind, offset = _header(wkb, 0)
    if kind == POLYGON:
        rings, _ = _polygon(wkb, offset, _order(wkb, 0), digits)
        return {"type": "Polygon", "coordinates": rings}
    if kind == MULTIPOLYGON:
        order = _order(wkb, 0)
        (count,) = struct.unpack_from(f"{order}I", wkb, offset)
        offset += 4
        polygons = []
        for _ in range(count):
            inner, offset = _header(wkb, offset)
            if inner != POLYGON:
                raise ValueError(f"A multipolygon holds WKB type {inner}, not a polygon")
            rings, offset = _polygon(wkb, offset, _order(wkb, offset - 5), digits)
            polygons.append(rings)
        return {"type": "MultiPolygon", "coordinates": polygons}
    raise ValueError(f"WKB type {kind} is not an area: only polygons and multipolygons are")


def feature_collection(areas: Iterable[tuple[str, str, bytes]]) -> dict[str, Any]:
    """(key, name, WKB) rows as a FeatureCollection whose features carry `id` and `name`."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": key, "name": name}, "geometry": geometry(wkb)}
            for key, name, wkb in areas
        ],
    }


def _order(wkb: bytes, offset: int) -> str:
    return "<" if wkb[offset] == 1 else ">"


def _header(wkb: bytes, offset: int) -> tuple[int, int]:
    """The geometry type at `offset`, and where its body starts."""
    (kind,) = struct.unpack_from(f"{_order(wkb, offset)}I", wkb, offset + 1)
    return kind, offset + 5


def _polygon(
    wkb: bytes, offset: int, order: str, digits: int
) -> tuple[list[list[list[float]]], int]:
    """A polygon's rings, and where the next geometry starts."""
    (count,) = struct.unpack_from(f"{order}I", wkb, offset)
    offset += 4
    rings = []
    for _ in range(count):
        (points,) = struct.unpack_from(f"{order}I", wkb, offset)
        offset += 4
        values = struct.unpack_from(f"{order}{2 * points}d", wkb, offset)
        offset += 16 * points
        rings.append(
            [
                [round(values[i], digits), round(values[i + 1], digits)]
                for i in range(0, len(values), 2)
            ]
        )
    return rings, offset
