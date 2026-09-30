import hashlib
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from mlops_core.config import DomainConfig
from mlops_core.data.extract import (
    CHECKS,
    MANIFEST_NAME,
    checks,
    checks_by_day,
    extract_all,
    find_link,
    ingest,
    ingest_file,
    ingestions,
    last_checked,
    latest_ingestion,
    user_agent,
)
from tests.conftest import PROFECO_CLOSED, PROFECO_FILE, PROFECO_PAGE
from tests.fakes import RecordedServer

T0 = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 12, 20, 12, 0, tzinfo=UTC)


def test_a_source_that_cannot_be_reached_does_not_stop_the_others(
    tmp_path: Path, coffee_config: DomainConfig, server: RecordedServer, client: httpx.Client
) -> None:
    """A government host that times out, or a page whose link moved: said, and the rest
    stored. SIAP's host did time out on a fresh clone's first extract."""
    del server.payloads[str(coffee_config.sources["cqi_2018"].url)]
    server.payloads[str(coffee_config.sources["world_bank_prices"].url)] = b"<html>moved</html>"

    extraction = extract_all(coffee_config, tmp_path, client)

    assert extraction.failed.keys() == {"cqi_2018", "world_bank_prices"}
    assert extraction.failed["cqi_2018"].startswith("HTTPStatusError: Client error '404")
    assert extraction.failed["world_bank_prices"].startswith("LookupError: No link on")
    assert "fred_usd_mxn" in extraction.artifacts  # after both in the config's order
    assert latest_ingestion(tmp_path, "cqi_2018") is None


def test_a_file_a_year_is_a_download_a_year_and_a_closed_year_is_kept(
    tmp_path: Path, coffee_config: DomainConfig, server: RecordedServer, client: httpx.Client
) -> None:
    """SIAP's closing statistics, one file a year: each its own partition. Asked again,
    only the last year is checked, and only when due; a year that fails is said, and the
    rest are stored."""
    source = coffee_config.sources["siap_agricola"]
    editions = source.editions()
    last = ingest("siap_agricola", source, tmp_path, client, now=T0)
    urls = [artifact.manifest.url for artifact in ingestions(tmp_path, "siap_agricola")]
    for _, url, _ in editions:
        server.payloads[url] = b"revised"
    soon = ingest("siap_agricola", source, tmp_path, client, now=T0 + timedelta(days=1))
    due = ingest("siap_agricola", source, tmp_path, client, now=T0 + timedelta(days=31))

    assert urls == [url for _, url, _ in editions]
    assert last.manifest.filename == "Cierre_agricola_mun_2025.csv"
    assert soon == last  # not due: nothing asked
    assert due.path.read_bytes() == b"revised" and due.manifest.url == editions[-1][1]
    assert len(ingestions(tmp_path, "siap_agricola")) == len(editions) + 1  # 2025 twice
    # A closed year is downloaded once, however often it is asked for.
    again = ingest("siap_agricola", source, tmp_path, client, now=T0 + timedelta(days=62))
    assert again == due and len(ingestions(tmp_path, "siap_agricola")) == len(editions) + 1


def test_a_year_that_fails_is_named_and_the_others_are_stored(
    tmp_path: Path, coffee_config: DomainConfig, server: RecordedServer, client: httpx.Client
) -> None:
    source = coffee_config.sources["siap_agricola"]
    year, url, _ = source.editions()[5]
    del server.payloads[url]

    extraction = extract_all(coffee_config, tmp_path, client)

    assert extraction.failed["siap_agricola"] == (
        f"LookupError: siap_agricola: not downloaded for {year} (HTTPStatusError); "
        "the rest is stored"
    )
    assert len(ingestions(tmp_path, "siap_agricola")) == len(source.editions()) - 1


def test_every_source_is_stored_byte_for_byte(
    tmp_path: Path,
    coffee_config: DomainConfig,
    client: httpx.Client,
    recorded: dict[str, bytes],
) -> None:
    extraction = extract_all(coffee_config, tmp_path, client)
    artifacts = extraction.artifacts

    assert not extraction.failed
    assert artifacts.keys() == recorded.keys()
    for name, artifact in artifacts.items():
        assert artifact.path.read_bytes() == recorded[name]
        assert artifact.manifest.sha256 == hashlib.sha256(recorded[name]).hexdigest()
        assert artifact.manifest.size_bytes == len(recorded[name])
        source = coffee_config.sources[name]
        if source.link is not None:  # the file the page links, which names the release
            assert re.search(source.link, artifact.manifest.url)
        elif source.link_text is not None:  # the one whose link says so
            assert artifact.manifest.url == {"profeco_prices": PROFECO_FILE,
                                              **PROFECO_CLOSED}[name]  # fmt: skip
        elif source.years is not None:  # the last year's, addressed by its year
            assert artifact.manifest.url == source.editions()[-1][1]
        else:
            assert artifact.manifest.url == str(source.url)
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
    assert len(list((tmp_path / "psd_coffee").glob("*=*"))) == 1


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


def test_a_file_is_found_by_what_its_link_says_when_its_address_says_nothing() -> None:
    """Tokens for addresses, markup and line breaks inside the text, and last year first:
    the pattern reads the text a person reads."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=PROFECO_PAGE))
    page = "https://datos.profeco.gob.mx/datos_abiertos/qqp.php"
    with httpx.Client(transport=transport) as client:
        found = find_link(client, page, r"^Quien es Quien en los Precios 2026$", on="text")
        anywhere = find_link(client, page, r"Precios \d{4}$", on="text")
        with pytest.raises(LookupError, match="update `link_text`"):
            find_link(client, page, r"Precios 2027", on="text")

    assert found == PROFECO_FILE
    assert anywhere.endswith("t=b954")  # the first match is the first listed, not the latest


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


def test_a_source_is_not_downloaded_again_until_it_is_due(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client, server: RecordedServer
) -> None:
    """Checked, not changed: an unchanged download stores nothing, and still counts."""
    source = coffee_config.sources["psd_coffee"].model_copy(update={"refresh_hours": 24})
    first = ingest("psd_coffee", source, tmp_path, client, now=T0)
    server.payloads[str(source.url)] = b"next circular"

    soon = ingest("psd_coffee", source, tmp_path, client, now=T0 + timedelta(hours=23))
    due = ingest("psd_coffee", source, tmp_path, client, now=T0 + timedelta(hours=25))

    assert soon == first  # not due: the new circular is not even asked for
    assert due.path.read_bytes() == b"next circular"
    assert last_checked(tmp_path, "psd_coffee") == T0 + timedelta(hours=25)


def test_the_check_is_recorded_when_nothing_changed(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client
) -> None:
    source = coffee_config.sources["psd_coffee"].model_copy(update={"refresh_hours": 24})
    first = ingest("psd_coffee", source, tmp_path, client, now=T0)
    again = ingest("psd_coffee", source, tmp_path, client, now=T0 + timedelta(hours=30))

    assert again == first  # unchanged: no new partition...
    assert last_checked(tmp_path, "psd_coffee") == T0 + timedelta(hours=30)  # ...but checked
    (tmp_path / "psd_coffee" / "checked_at").unlink()  # ingested before checks were kept
    assert last_checked(tmp_path, "psd_coffee") == T0
    assert last_checked(tmp_path, "never_ingested") is None


def test_a_document_is_not_fetched_again_until_it_is_due(
    tmp_path: Path, client: httpx.Client, server: RecordedServer
) -> None:
    """The corpus goes through `ingest_file`, with the corpus's own `refresh_hours`."""
    url = "https://publisher.test/paper.pdf"
    server.payloads[url] = b"%PDF first edition"
    first = ingest_file("paper", url, "paper.pdf", tmp_path, client, T0, refresh_hours=720)
    server.payloads[url] = b"%PDF revised"

    ten_days = T0 + timedelta(days=10)
    soon = ingest_file("paper", url, "paper.pdf", tmp_path, client, ten_days, refresh_hours=720)

    assert soon == first and soon.path.read_bytes() == b"%PDF first edition"


def test_every_download_is_logged_changed_or_not(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client
) -> None:
    """An unchanged download stores nothing, but the day it happened is still history."""
    source = coffee_config.sources["psd_coffee"]
    first = ingest("psd_coffee", source, tmp_path, client, now=T0)
    ingest("psd_coffee", source, tmp_path, client, now=T1)  # the same bytes

    logged = checks(tmp_path, "psd_coffee")

    assert [(c.checked_at, c.partition, c.changed) for c in logged] == [
        (T0, first.partition.name, True),
        (T1, first.partition.name, False),  # found what T0 left
    ]
    assert len((tmp_path / "psd_coffee" / CHECKS).read_text().splitlines()) == 2


def test_a_source_read_before_the_log_tells_its_changes_and_its_last_check(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client
) -> None:
    source = coffee_config.sources["psd_coffee"]
    first = ingest("psd_coffee", source, tmp_path, client, now=T0)
    ingest("psd_coffee", source, tmp_path, client, now=T1)
    (tmp_path / "psd_coffee" / CHECKS).unlink()  # as a source ingested before the log

    assert [(c.checked_at, c.changed) for c in checks(tmp_path, "psd_coffee")] == [
        (T0, True),  # its partition
        (T1, False),  # its `checked_at`, which found that partition
    ]
    assert first.partition.name == checks(tmp_path, "psd_coffee")[-1].partition
    assert checks(tmp_path, "never_read") == []


def test_downloads_are_grouped_by_the_day_they_were_made_in_the_operators_calendar(
    tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client
) -> None:
    """20:30 in Mexico City is already tomorrow in UTC: the day is the operator's."""
    source = coffee_config.sources["psd_coffee"]
    evening = datetime(2026, 9, 28, 2, 30, tzinfo=UTC)  # 27 September, 20:30 in the city
    ingest("psd_coffee", source, tmp_path, client, now=evening)

    assert list(checks_by_day(tmp_path, "psd_coffee", "America/Mexico_City")) == ["2026-09-27"]
    assert list(checks_by_day(tmp_path, "psd_coffee", "UTC")) == ["2026-09-28"]
