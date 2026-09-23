"""The corpus: what is fetched, what a person hands over, and what comes out as text.

The fixtures are written rather than recorded: the real corpus is copyrighted and 62 MB,
while what the reader has to get right - page numbering across an empty page, a file
encrypted with permissions, an article's sections - fits in a few hundred bytes.
"""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from mlops_core.config import DocumentConfig
from mlops_core.contracts import check_contract
from mlops_core.data.documents import (
    DOCUMENT_PARTS,
    fetch_documents,
    inbox_dir,
    read_document,
)
from mlops_core.data.extract import RawArtifact, latest_ingestion, store_payload

FIXTURES = Path(__file__).parent / "fixtures" / "documents"
NOW = datetime(2026, 9, 23, tzinfo=UTC)


def document(name: str, **values: object) -> DocumentConfig:
    fields: dict[str, object] = {
        "name": name,
        "title": "How coffee is processed",
        "publisher": "A publisher",
        "year": 2020,
        "url": "https://publisher.test/paper",
        "license": "CC BY",
        "language": "en",
        "topics": ["processing"],
        "format": "pdf",
    }
    return DocumentConfig.model_validate(fields | values)


def stored(fixture: str, name: str, raw_dir: Path, **values: object) -> RawArtifact:
    """The fixture in a raw partition, as an ingestion leaves it."""
    config = document(name, **values)
    return store_payload(
        name, fixture, (FIXTURES / fixture).read_bytes(), raw_dir, str(config.url), NOW
    )


def test_a_pdf_is_read_page_by_page_and_keeps_its_page_numbers(tmp_path: Path) -> None:
    """The middle page has no text. Renumbering what is left would make a citation point
    at the wrong page."""
    config = document("sample")
    artifact = stored("sample.pdf", "sample", tmp_path)

    parts = check_contract(DOCUMENT_PARTS, read_document(artifact, config))

    assert parts["part"].to_list() == [1, 3]
    assert parts["text"].to_list() == ["Coffee is the seed of a fruit.", "Altitude shapes acidity."]
    assert parts["part_title"].null_count() == parts.height  # a PDF has no headings to trust


def test_a_file_encrypted_with_permissions_is_opened(tmp_path: Path) -> None:
    """The SCA's standards are encrypted, not password-protected: an empty one opens them."""
    config = document("secured")
    artifact = stored("secured.pdf", "secured", tmp_path)

    parts = read_document(artifact, config)

    assert parts["text"][0] == "Coffee is the seed of a fruit."


def test_an_article_is_read_by_section_with_its_headings(tmp_path: Path) -> None:
    """Top-level sections only: a nested section's text is inside its parent's, so
    emitting both would index every paragraph twice."""
    config = document("article", format="jats")
    artifact = stored("article.xml", "article", tmp_path, format="jats")

    parts = check_contract(DOCUMENT_PARTS, read_document(artifact, config))

    assert parts["part_title"].to_list() == ["Abstract", "Introduction", "Results"]
    introduction = parts.filter(parts["part_title"] == "Introduction")["text"].item()
    assert "Only the wet mill is covered." in introduction  # the nested section, once
    assert introduction.count("Only the wet mill") == 1
    # The ligature a PDF or a publisher writes is normalised, or no search matches it.
    assert "Coffee cherries ferment" in introduction


def test_a_document_with_no_text_is_refused(tmp_path: Path) -> None:
    """Almost always a scan. Silence would index an empty document and answer nothing."""
    blank = tmp_path / "blank.pdf"
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with blank.open("wb") as out:
        writer.write(out)
    artifact = store_payload("blank", "blank.pdf", blank.read_bytes(), tmp_path, "https://x.test")

    with pytest.raises(ValueError, match="No text could be read"):
        read_document(artifact, document("blank"))


def transport(bodies: dict[str, bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = bodies.get(str(request.url))
        return httpx.Response(200, content=body) if body else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_document_a_publisher_serves_is_fetched(tmp_path: Path) -> None:
    served = document("served", url="https://publisher.test/open.pdf")
    client = transport({str(served.url): (FIXTURES / "sample.pdf").read_bytes()})

    artifacts, missing = fetch_documents([served], tmp_path, client, NOW)

    assert missing == {}
    manifest = artifacts["served"].manifest
    assert manifest.url == str(served.url) and manifest.size_bytes > 0
    assert artifacts["served"].path.name == "served.pdf"


def test_a_document_behind_a_refusal_waits_in_the_inbox(tmp_path: Path) -> None:
    """Fetched by hand and dropped in; nothing is scraped around the 403."""
    handed = document("handed", inbox="paper.pdf")
    client = transport({})

    _, missing = fetch_documents([handed], tmp_path, client, NOW)
    assert "not in the inbox" in missing["handed"]
    assert str(handed.url) in missing["handed"]

    inbox = inbox_dir(tmp_path)
    inbox.mkdir(parents=True)
    (inbox / "paper.pdf").write_bytes((FIXTURES / "sample.pdf").read_bytes())
    artifacts, missing = fetch_documents([handed], tmp_path, client, NOW)

    assert missing == {}
    # The manifest keeps the URL it came from, not the path it was handed over at: that
    # is what an answer cites.
    assert artifacts["handed"].manifest.url == str(handed.url)
    assert latest_ingestion(tmp_path / "raw", "handed") is not None


def test_an_article_is_stored_as_xml_not_as_pdf(tmp_path: Path) -> None:
    article = document("article", format="jats", url="https://publisher.test/fullTextXML")
    client = transport({str(article.url): (FIXTURES / "article.xml").read_bytes()})

    artifacts, _ = fetch_documents([article], tmp_path, client, NOW)

    assert artifacts["article"].path.name == "article.xml"
