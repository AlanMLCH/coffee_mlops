import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from mlops_core.config import DomainConfig
from mlops_core.data.extract import (
    MANIFEST_NAME,
    extract_all,
    find_link,
    ingest,
    ingestions,
    latest_ingestion,
    user_agent,
)
from tests.fakes import RecordedServer

T0 = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 12, 20, 12, 0, tzinfo=UTC)


def test_every_source_is_stored_byte_for_byte(
    tmp_path: Path,
    coffee_config: DomainConfig,
    client: httpx.Client,
    recorded: dict[str, bytes],
) -> None:
    artifacts = extract_all(coffee_config, tmp_path, client)

    assert artifacts.keys() == recorded.keys()
    for name, artifact in artifacts.items():
        assert artifact.path.read_bytes() == recorded[name]
        assert artifact.manifest.sha256 == hashlib.sha256(recorded[name]).hexdigest()
        assert artifact.manifest.size_bytes == len(recorded[name])
        source = coffee_config.sources[name]
        if source.link is None:
            assert artifact.manifest.url == str(source.url)
        else:  # the file the page links, which names the release
            assert re.search(source.link, artifact.manifest.url)
        assert artifact.partition.name.startswith("ingested_at=")


def test_signed_redirect_url_is_never_persisted(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client
) -> None:
    artifact = ingest("cqi_2023", coffee_config.sources["cqi_2023"], tmp_path, client)

    assert "Signature" not in (artifact.partition / MANIFEST_NAME).read_text()


def test_unchanged_upstream_is_not_stored_again(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client
) -> None:
    source = coffee_config.sources["psd_coffee"]
    first = ingest("psd_coffee", source, tmp_path, client, now=T0)
    second = ingest("psd_coffee", source, tmp_path, client, now=T1)

    assert second == first
    assert len(list((tmp_path / "psd_coffee").iterdir())) == 1


def test_changed_upstream_creates_a_new_partition(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client, server: RecordedServer
) -> None:
    source = coffee_config.sources["psd_coffee"]
    first = ingest("psd_coffee", source, tmp_path, client, now=T0)
    server.payloads[str(source.url)] = b"new circular"
    second = ingest("psd_coffee", source, tmp_path, client, now=T1)

    assert second.partition != first.partition
    assert first.path.is_file()  # history is kept
    assert latest_ingestion(tmp_path, "psd_coffee") == second


@pytest.mark.parametrize("status", [404, 500])
def test_http_error_leaves_no_partial_files(
    tmp_path: Path, coffee_config: DomainConfig, status: int
) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(status))
    with httpx.Client(transport=transport) as failing, pytest.raises(httpx.HTTPStatusError):
        ingest("psd_coffee", coffee_config.sources["psd_coffee"], tmp_path, failing)

    assert list((tmp_path / "psd_coffee").iterdir()) == []


def test_empty_body_is_rejected(tmp_path: Path, coffee_config: DomainConfig) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b""))
    with httpx.Client(transport=transport) as empty, pytest.raises(ValueError, match="Empty"):
        ingest("psd_coffee", coffee_config.sources["psd_coffee"], tmp_path, empty)

    assert list((tmp_path / "psd_coffee").iterdir()) == []


def test_partition_without_manifest_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "psd_coffee" / "ingested_at=20260919T120000Z").mkdir(parents=True)

    assert latest_ingestion(tmp_path, "psd_coffee") is None


def test_the_user_agent_says_who_is_calling_and_how_to_reach_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read from the package metadata, not written into the core: the core does not know
    which project ships it."""
    user_agent.cache_clear()
    assert user_agent().endswith("(+https://github.com/AlanMLCH/coffee_mlops)")

    user_agent.cache_clear()
    monkeypatch.setattr("mlops_core.data.extract.distributions", lambda: [])
    assert user_agent() == "mlops_core"  # a bare source tree still says what is calling
    user_agent.cache_clear()


# --- A file found by its link, a host that drops connections, a history of windows ------


def test_a_file_is_found_by_the_link_its_page_gives_it() -> None:
    page = '<a href="/data/prices-2026-08.xlsx">Monthly</a><a href="notes.pdf">Notes</a>'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=page))
    with httpx.Client(transport=transport) as client:
        found = find_link(client, "https://bank.test/research/prices", r"prices-.*\.xlsx$")
        with pytest.raises(LookupError, match="No link on https://bank"):
            find_link(client, "https://bank.test/research/prices", r"\.csv$")

    assert found == "https://bank.test/data/prices-2026-08.xlsx"  # relative, made absolute


def test_a_dropped_connection_is_tried_again_and_an_http_error_is_not(
    tmp_path: Path, coffee_config: DomainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr("mlops_core.data.extract.time.sleep", waits.append)
    calls: list[str] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) < 3:
            raise httpx.ConnectError("connection reset by peer")
        return httpx.Response(200, content=b"circular")

    source = coffee_config.sources["psd_coffee"]
    with httpx.Client(transport=httpx.MockTransport(flaky)) as client:
        artifact = ingest("psd_coffee", source, tmp_path, client)

    assert artifact.path.read_bytes() == b"circular"
    assert (len(calls), waits) == (3, [1, 2])

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection reset by peer")

    with (
        httpx.Client(transport=httpx.MockTransport(down)) as client,
        pytest.raises(httpx.ConnectError),
    ):
        ingest("psd_coffee", source, tmp_path / "down", client)
    assert waits == [1, 2, 1, 2, 4]  # four attempts, then it says so
    assert list((tmp_path / "down" / "psd_coffee").iterdir()) == []


def test_every_ingestion_of_a_source_is_its_history(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client, server: RecordedServer
) -> None:
    source = coffee_config.sources["psd_coffee"]
    first = ingest("psd_coffee", source, tmp_path, client, now=T0)
    server.payloads[str(source.url)] = b"next month"
    second = ingest("psd_coffee", source, tmp_path, client, now=T1)
    (tmp_path / "psd_coffee" / "ingested_at=20991231T000000Z").mkdir()  # interrupted

    assert ingestions(tmp_path, "psd_coffee") == [first, second]
    assert ingestions(tmp_path, "never_ingested") == []
