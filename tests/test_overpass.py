"""Overpass answers one query rather than a page, and it fails inside a 200.

Those two facts drive the tests: an answer that came back short must stop the run
instead of being stored as a smaller inventory, and the elements that carry no
coordinate of their own (cafes mapped as buildings) must survive the trip.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from mlops_core.config import OverpassConfig
from mlops_core.data.api import ApiClient
from mlops_core.data.sources.overpass import COUNT, build_query, fetch, ingest_places
from mlops_core.storage import MANIFEST_NAME

# A real answer, trimmed: three cafes mapped as nodes and two mapped as buildings.
FIXTURE = Path(__file__).parent / "fixtures" / "overpass_cafes_sample.json"
CONFIG = OverpassConfig(
    name="osm_cafes",
    base_url="https://overpass.test/api/interpreter",
    area_iso="MX-CMX",
    amenity="cafe",
    filename="osm_cafes.json",
    timeout_s=50,
    rate_limit_seconds=0.0,
    cache_hours=24,
)


def recorded() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload


class FakeOverpass:
    """Replays a payload and remembers the queries it was asked."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = recorded() if payload is None else payload
        self.queries: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.queries.append(request.url.params["data"])
        return httpx.Response(200, json=self.payload)


def build_client(service: FakeOverpass, cache_dir: Path) -> ApiClient:
    return ApiClient(
        client=httpx.Client(transport=httpx.MockTransport(service.handler)),
        cache_dir=cache_dir,
        min_interval_s=0.0,
        sleep=lambda _: None,
    )


def test_the_query_asks_for_every_element_type_inside_the_area() -> None:
    query = build_query(CONFIG)

    assert "[out:json][timeout:50]" in query
    assert 'area["ISO3166-2"="MX-CMX"]' in query
    for element_type in ("node", "way", "relation"):
        assert f'{element_type}["amenity"="cafe"](area.a)' in query
    # Without `center`, the cafes mapped as buildings come back with no coordinate.
    assert query.endswith("out center tags;")


def test_the_count_form_asks_the_same_question_without_the_answer() -> None:
    """The live check wants to know the selectors still match, not the inventory."""
    counting = build_query(CONFIG, COUNT)

    assert counting.endswith("out count;")
    assert counting.removesuffix("out count;") == build_query(CONFIG).removesuffix(
        "out center tags;"
    )


def test_places_mapped_as_buildings_keep_the_centre(tmp_path: Path) -> None:
    service = FakeOverpass()

    artifact = ingest_places(build_client(service, tmp_path / "cache"), CONFIG, tmp_path / "raw")

    stored = json.loads(artifact.path.read_text(encoding="utf-8"))
    ways = [e for e in stored["elements"] if e["type"] == "way"]
    assert len(stored["elements"]) == 5  # nothing dropped on the way in
    assert ways and all("center" in way for way in ways)


def test_a_partial_answer_stops_the_run(tmp_path: Path) -> None:
    """Overpass reports its own failures inside a 200; stored, it would look complete."""
    service = FakeOverpass({"elements": [], "remark": "runtime error: Query timed out"})

    with pytest.raises(RuntimeError, match="timed out"):
        fetch(build_client(service, tmp_path / "cache"), CONFIG)


def test_an_element_without_a_coordinate_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    payload = recorded()
    payload["elements"][0] = {"type": "node", "id": 1, "tags": {"amenity": "cafe"}}

    ingest_places(build_client(FakeOverpass(payload), tmp_path / "cache"), CONFIG, tmp_path / "raw")

    assert "1 of 5 elements came back without a coordinate" in caplog.text


def test_the_licence_and_the_query_are_stored_with_the_data(tmp_path: Path) -> None:
    """ODbL asks for the attribution to travel with the data; the query explains it."""
    service = FakeOverpass()

    artifact = ingest_places(build_client(service, tmp_path / "cache"), CONFIG, tmp_path / "raw")

    stored = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert "ODbL" in stored["license"]
    assert stored["query"] == build_query(CONFIG)
    assert artifact.manifest.url == CONFIG.base_url
    assert MANIFEST_NAME in {path.name for path in artifact.partition.iterdir()}


def test_the_same_places_in_another_order_are_not_new_data(tmp_path: Path) -> None:
    """The database is live and the answer's timestamp moves every minute: only the
    elements decide whether this pull is different from the last one."""
    raw = tmp_path / "raw"
    ingest_places(build_client(FakeOverpass(), tmp_path / "a"), CONFIG, raw)

    shuffled = recorded()
    shuffled["elements"].reverse()
    shuffled["osm3s"]["timestamp_osm_base"] = "2026-09-21T09:00:00Z"
    second = ingest_places(build_client(FakeOverpass(shuffled), tmp_path / "b"), CONFIG, raw)

    assert len(list((raw / "osm_cafes").iterdir())) == 1  # one partition, not two
    assert json.loads(second.path.read_text(encoding="utf-8"))["elements"][0]["type"] == "node"


def test_re_running_hits_the_cache_instead_of_the_service(tmp_path: Path) -> None:
    service = FakeOverpass()
    client = build_client(service, tmp_path / "cache")
    ingest_places(client, CONFIG, tmp_path / "raw")

    ingest_places(client, CONFIG, tmp_path / "raw")

    assert len(service.queries) == 1  # the second run went no further than the disk


def test_a_different_query_is_a_different_cache_entry(tmp_path: Path) -> None:
    """The cache key is the query itself: changing the area must not replay the old area."""
    service = FakeOverpass()
    client = build_client(service, tmp_path / "cache")
    fetch(client, CONFIG)

    fetch(client, CONFIG.model_copy(update={"area_iso": "MX-JAL"}))

    assert len(service.queries) == 2
