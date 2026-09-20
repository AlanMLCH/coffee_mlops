"""DENUE pages 100 records at a time and puts the token in the URL path.

Both facts drive the tests: a short page must end the walk without truncating silently,
and the credential must not survive into a cache key, a manifest or a log line.
"""

import json
import logging
from pathlib import Path

import httpx
import pytest

from coffee_mlops.config import DenueConfig
from coffee_mlops.data.api import ApiClient
from coffee_mlops.data.sources.denue import (
    DOCUMENTED_URL,
    count,
    establishments,
    ingest_establishments,
)
from coffee_mlops.storage import MANIFEST_NAME

TOKEN = "super-secret-token"
CONFIG = DenueConfig(
    name="denue_cafes",
    base_url="https://denue.test/consulta",
    entity="09",
    activity_class="722515",
    page_size=100,
    filename="denue_cafes.json",
    rate_limit_seconds=0.0,
)


class FakeDenue:
    """Answers like the real service: a count, and pages of at most `page_size`."""

    def __init__(self, total: int, reported_total: int | None = None) -> None:
        self.total = total
        self.reported_total = total if reported_total is None else reported_total
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(path)
        if "/Cuantificar/" in path:
            return httpx.Response(
                200, json=[{"AE": "722515", "AG": "09", "Total": str(self.reported_total)}]
            )
        start, end = (int(part) for part in path.split("/")[-4:-2])
        rows = [
            {"Id": str(i), "Nombre": f"CAFE {i}", "Latitud": "19.4", "Longitud": "-99.1"}
            for i in range(start, min(end, self.total) + 1)
        ]
        return httpx.Response(200, json=rows)


def build_client(service: FakeDenue, tmp_path: Path) -> ApiClient:
    return ApiClient(
        client=httpx.Client(transport=httpx.MockTransport(service.handler)),
        cache_dir=tmp_path / "cache",
        min_interval_s=0.0,
        sleep=lambda _: None,
    )


def test_the_count_comes_before_the_download(tmp_path: Path) -> None:
    service = FakeDenue(total=250)

    assert count(build_client(service, tmp_path), CONFIG, TOKEN) == 250
    assert len(service.requests) == 1  # one cheap call, no records fetched


def test_every_page_is_walked_until_a_short_one(tmp_path: Path) -> None:
    service = FakeDenue(total=250)

    records = list(establishments(build_client(service, tmp_path), CONFIG, TOKEN))

    assert len(records) == 250
    assert records[0]["Id"] == "1" and records[-1]["Id"] == "250"
    assert len(service.requests) == 3  # 1-100, 101-200, 201-300 (which came back short)


def test_an_exact_multiple_of_the_page_size_still_terminates(tmp_path: Path) -> None:
    """The walk stops on a short page, so a total of exactly 200 needs one empty page."""
    service = FakeDenue(total=200)

    records = list(establishments(build_client(service, tmp_path), CONFIG, TOKEN))

    assert len(records) == 200
    assert len(service.requests) == 3


def test_the_inventory_is_stored_without_the_token(tmp_path: Path) -> None:
    service = FakeDenue(total=120)
    client = build_client(service, tmp_path)

    artifact = ingest_establishments(client, CONFIG, TOKEN, tmp_path / "raw")

    records = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert len(records) == 120
    assert artifact.manifest.url == DOCUMENTED_URL
    assert TOKEN not in (artifact.partition / MANIFEST_NAME).read_text()
    assert all(TOKEN not in path.name for path in (tmp_path / "cache").iterdir())


def test_a_page_short_of_the_reported_count_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A truncated download looks exactly like a complete one; only the count knows."""
    caplog.set_level(logging.WARNING)
    service = FakeDenue(total=150, reported_total=400)

    ingest_establishments(build_client(service, tmp_path), CONFIG, TOKEN, tmp_path / "raw")

    assert "returned 150 records but reported 400" in caplog.text


def test_re_running_hits_the_cache_instead_of_the_service(tmp_path: Path) -> None:
    service = FakeDenue(total=120)
    client = build_client(service, tmp_path)
    ingest_establishments(client, CONFIG, TOKEN, tmp_path / "raw")
    calls = len(service.requests)

    ingest_establishments(client, CONFIG, TOKEN, tmp_path / "raw")

    assert len(service.requests) == calls  # nothing went out a second time


def test_the_same_inventory_in_another_order_is_not_new_data(tmp_path: Path) -> None:
    """A live register hands its pages back in a different order each time; only a
    canonical form keeps that from looking like a change."""
    raw = tmp_path / "raw"
    forwards = FakeDenue(total=120)
    ingest_establishments(build_client(forwards, tmp_path / "a"), CONFIG, TOKEN, raw)

    backwards = FakeDenue(total=120)
    backwards.handler = _reversing(backwards.handler)  # type: ignore[method-assign]
    second = ingest_establishments(build_client(backwards, tmp_path / "b"), CONFIG, TOKEN, raw)

    assert len(list((raw / "denue_cafes").iterdir())) == 1  # one partition, not two
    assert json.loads(second.path.read_text(encoding="utf-8"))[0]["Id"] == "1"


def _reversing(handler: object) -> object:
    def reversed_handler(request: httpx.Request) -> httpx.Response:
        response = handler(request)  # type: ignore[operator]
        rows = response.json()
        return httpx.Response(200, json=list(reversed(rows)) if isinstance(rows, list) else rows)

    return reversed_handler
