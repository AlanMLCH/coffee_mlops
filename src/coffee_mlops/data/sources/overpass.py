"""OpenStreetMap: every place tagged with one amenity inside one administrative area.

Overpass is the query endpoint over OSM's live database. It needs no credential, which
is exactly why it is the primary geolocated source for a public repository: Google
Places would give richer attributes, but its terms forbid storing them.

Three things about the service shape this module:

- **Ways carry no coordinate of their own.** A cafe mapped as a building is a closed
  way, so `out center` is what makes those 94 of 1,125 elements usable next to the
  1,031 nodes instead of silently dropped.
- **A query that runs out of time answers 200.** The body then carries a `remark`
  instead of (or beside) the elements, so a partial inventory looks like a complete
  one. It is checked and raised, never stored.
- **It is run by volunteers.** One query per run, rate limited, with the retries in
  `ApiClient` covering the 429 the instance answers when it is busy.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from coffee_mlops.config import OverpassConfig
from coffee_mlops.data.api import ApiClient
from coffee_mlops.data.extract import RawArtifact, store_payload

logger = logging.getLogger(__name__)

# Every element the query returns, with a representative point for the ones that are
# not a single node. The count form answers "does this still work?" without shipping
# the inventory, which is what the live check uses.
ELEMENTS = "center tags"
COUNT = "count"


def build_query(config: OverpassConfig, output: str = ELEMENTS) -> str:
    """Overpass QL asking for one amenity in one area, as nodes, ways and relations.

    Relations are in the union although Mexico City currently has none tagged
    `amenity=cafe`: a cafe mapped as a multipolygon is legal OSM, and leaving the type
    out would lose it without a word.
    """
    selector = f'["amenity"="{config.amenity}"]'
    return (
        f"[out:json][timeout:{config.timeout_s}];"
        f'area["ISO3166-2"="{config.area_iso}"]->.a;'
        f"(node{selector}(area.a);way{selector}(area.a);relation{selector}(area.a););"
        f"out {output};"
    )


def fetch(client: ApiClient, config: OverpassConfig, output: str = ELEMENTS) -> dict[str, Any]:
    """Run one Overpass query and return the parsed body, refusing a partial answer.

    The whole query goes into the cache key: it is the request's entire meaning, and
    unlike DENUE's URL it carries no credential, so hashing it is safe. Keying on
    anything shorter would replay yesterday's answer after the query changed.
    """
    query = build_query(config, output)
    payload: dict[str, Any] = client.get_json(
        f"{config.base_url}?data={quote(query)}", cache_key=query
    )
    remark = payload.get("remark")
    if remark is not None:
        # Overpass reports its own failures inside a 200: timeout, out of memory, a
        # syntax error in the query. Storing that body would record an inventory that
        # is short for a reason nothing downstream could see.
        raise RuntimeError(f"Overpass refused the query: {remark}")
    return payload


def ingest_places(
    client: ApiClient,
    config: OverpassConfig,
    raw_dir: Path,
    now: datetime | None = None,
) -> RawArtifact:
    """Store the whole inventory as one raw JSON document, in a canonical order."""
    payload = fetch(client, config)
    elements = list(payload["elements"])
    without_coordinates = [e for e in elements if "lat" not in e and "center" not in e]
    if without_coordinates:
        # Not fatal -- the raw layer stores what the service said -- but a place with no
        # point is unusable for the spatial join, so it must not pass unnoticed.
        logger.warning(
            "Overpass: %d of %d elements came back without a coordinate",
            len(without_coordinates),
            len(elements),
        )
    logger.info("Overpass: %d elements tagged amenity=%s", len(elements), config.amenity)

    # Same reason as DENUE: the service is free to answer in any order, and without a
    # canonical one every run would hash differently and store an identical partition.
    elements.sort(key=lambda element: (element["type"], element["id"]))
    document = {
        # ODbL requires the attribution to travel with the data, so it is stored with it.
        "license": payload["osm3s"]["copyright"],
        # What produced these rows, for anyone reading the file two stages from now.
        "query": build_query(config),
        "elements": elements,
    }
    # `osm3s.timestamp_osm_base` is deliberately left out: it moves every minute, so
    # keeping it would defeat the de-duplication and fill raw/ with identical pulls.
    # The manifest's `ingested_at` already dates the pull.
    body = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return store_payload(config.name, config.filename, body, raw_dir, config.base_url, now)
