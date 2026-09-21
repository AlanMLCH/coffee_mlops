"""FAS answers by year, with ids instead of names, and wants its key in a header.

Those three facts drive the tests: the walk must cover every market year up to the
forecast one, the ids must decode into exactly the PSD file's columns and pass its
contract, and the key must never reach a URL, a cache file name or a manifest.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandera.errors
import pytest

from coffee_mlops.config import FasConfig
from coffee_mlops.contracts import check_contract
from coffee_mlops.data.api import ApiClient
from coffee_mlops.data.schemas import PSD_COFFEE
from coffee_mlops.data.sources.fas import KEY_HEADER, fetch_rows, ingest_balance, to_frame
from coffee_mlops.storage import MANIFEST_NAME
from tests.fakes import fas_recording, fas_response

KEY = "super-secret-key"
NOW = datetime(2026, 9, 21, tzinfo=UTC)
CONFIG = FasConfig(
    name="fas_psd_coffee",
    base_url="https://api.fas.usda.gov/api/psd",
    commodity_code="0711100",
    first_year=2021,  # the recording holds 2022 and 2023; 2021 and 2024+ come back empty
    filename="fas_psd_coffee.json",
    rate_limit_seconds=0.0,
    cache_hours=24,
)


class FakeFas:
    """Replays the recording, remembers every request, and refuses one without the key."""

    def __init__(self, recording: dict[str, Any] | None = None) -> None:
        self.recording = fas_recording() if recording is None else recording
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = fas_response(request, self.recording)
        assert response is not None
        return response


def build_client(service: FakeFas, cache_dir: Path) -> ApiClient:
    return ApiClient(
        client=httpx.Client(transport=httpx.MockTransport(service.handler)),
        cache_dir=cache_dir,
        min_interval_s=0.0,
        sleep=lambda _: None,
    )


def test_every_market_year_is_asked_up_to_the_forecast_one(tmp_path: Path) -> None:
    """PSD files next season's forecast under the next year; beyond it the list is empty."""
    service = FakeFas()

    rows = list(fetch_rows(build_client(service, tmp_path), CONFIG, KEY, last_year=2027))

    asked = [r.url.path.rsplit("/", 1)[1] for r in service.requests]
    assert asked == [str(year) for year in range(2021, 2028)]
    assert {row["marketYear"] for row in rows} == {"2022", "2023"}
    assert len(rows) == 114


def test_the_key_travels_in_a_header_and_nowhere_else(tmp_path: Path) -> None:
    service = FakeFas()

    artifact = ingest_balance(
        build_client(service, tmp_path / "cache"), CONFIG, KEY, tmp_path / "raw", NOW
    )

    assert all(request.headers[KEY_HEADER] == KEY for request in service.requests)
    assert all(KEY not in str(request.url) for request in service.requests)
    assert KEY not in (artifact.partition / MANIFEST_NAME).read_text()
    assert all(KEY not in path.name for path in tmp_path.rglob("*"))


def test_the_rows_are_stored_with_the_lookups_that_decode_them(tmp_path: Path) -> None:
    """Reading the raw layer back must not depend on asking the service again."""
    artifact = ingest_balance(
        build_client(FakeFas(), tmp_path / "cache"), CONFIG, KEY, tmp_path / "raw", NOW
    )

    stored = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert set(stored) == {"commodities", "attributes", "units", "countries", "rows"}
    assert len(stored["rows"]) == 114


def test_decoded_rows_have_the_files_columns_and_pass_its_contract() -> None:
    """One table, two roads: the API is held to exactly the PSD file's schema."""
    frame = check_contract(PSD_COFFEE, to_frame(documented()))

    assert frame.height == 114
    # The API pads unit descriptions to a fixed width; the file does not.
    assert frame["Unit_Description"].unique().to_list() == ["(1000 60 KG BAGS)"]
    assert set(frame["Country_Name"]) == {"Brazil", "Colombia", "Mexico"}


def test_an_id_its_lookup_does_not_know_is_refused_rather_than_guessed() -> None:
    document = documented()
    document["countries"] = [c for c in document["countries"] if c["countryCode"] != "MX"]

    with pytest.raises(pandera.errors.SchemaErrors, match="Country_Name"):
        check_contract(PSD_COFFEE, to_frame(document))


def test_an_empty_pull_is_an_error_not_an_empty_market(tmp_path: Path) -> None:
    empty = fas_recording() | {"years": {}}

    with pytest.raises(RuntimeError, match="no rows"):
        ingest_balance(build_client(FakeFas(empty), tmp_path), CONFIG, KEY, tmp_path, NOW)


def test_the_same_answer_in_another_order_is_not_new_data(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    ingest_balance(build_client(FakeFas(), tmp_path / "a"), CONFIG, KEY, raw, NOW)
    shuffled = fas_recording()
    shuffled["years"] = {year: rows[::-1] for year, rows in shuffled["years"].items()}
    shuffled["countries"] = shuffled["countries"][::-1]

    ingest_balance(build_client(FakeFas(shuffled), tmp_path / "b"), CONFIG, KEY, raw, NOW)

    assert len(list((raw / "fas_psd_coffee").iterdir())) == 1  # one partition, not two


def documented() -> dict[str, Any]:
    """The recording in the shape `ingest_balance` stores it."""
    recording = fas_recording()
    rows = [row for year in sorted(recording["years"]) for row in recording["years"][year]]
    return {**{k: v for k, v in recording.items() if k != "years"}, "rows": rows}
