"""The corpus: documents the agent explains from, ingested like any other source.

A document is text, not figures. What a variety is, how a process changes a cup, what an
attribute of a cupping form means: that is what a document answers. Numbers are answered
from the tables, because a PDF's tables come out of text extraction scrambled and an
answer built from them is confidently wrong.

Publishers differ in whether a robot may fetch them: some serve the file to anyone, some
answer 403 to everything that is not a browser, licence notwithstanding. The second kind
is fetched by hand into the inbox and named in the config with the URL it came from - a
refusal is respected, never worked around. Either way the bytes land in `raw/` with the
same manifest, de-duplication and atomicity as a CSV, and every part carries the document
it came from, so a chunk can later say who published it and when.
"""

import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandera.polars as pa
import polars as pl

from mlops_core.config import DocumentConfig
from mlops_core.data.extract import RawArtifact, ingest_file, store_payload

logger = logging.getLogger(__name__)

INBOX = ("inbox", "documents")  # under the domain's data dir

# One row per part: a page of a PDF, a section of an article. Splitting into chunks comes
# later and needs the parts, because a heading is what a chunk is given for context.
DOCUMENT_PARTS = pa.DataFrameSchema(
    name="document_parts",
    strict=True,
    unique=["document_id", "part"],
    columns={
        "document_id": pa.Column(pl.String),
        "part": pa.Column(pl.Int64, pa.Check.ge(1)),
        "part_title": pa.Column(pl.String, nullable=True),
        "text": pa.Column(pl.String, pa.Check.str_length(min_value=1)),
    },
)


def inbox_dir(data_dir: Path) -> Path:
    return data_dir.joinpath(*INBOX)


def fetch_documents(
    documents: list[DocumentConfig],
    data_dir: Path,
    client: httpx.Client,
    now: datetime | None = None,
) -> tuple[dict[str, RawArtifact], dict[str, str]]:
    """Every document into `raw/`, and the ones nobody handed over yet, with what to do."""
    artifacts, missing = {}, {}
    for document in documents:
        try:
            artifacts[document.name] = _fetch(document, data_dir, client, now)
        except FileNotFoundError as absent:
            missing[document.name] = str(absent)
    return artifacts, missing


def read_document(artifact: RawArtifact, document: DocumentConfig) -> pl.DataFrame:
    """The stored file as text, one row per part, in the order it is read.

    Nothing is cleaned here - this is the raw layer's reader, the counterpart of parsing
    a CSV - except the Unicode normalisation that turns a PDF's ligatures ("coﬀee") into
    the letters a search can match.
    """
    parts = _pdf_parts(artifact.path) if document.format == "pdf" else _jats_parts(artifact.path)
    rows = [
        {
            "document_id": document.name,
            "part": number,
            "part_title": title,
            "text": unicodedata.normalize("NFKC", text).strip(),
        }
        for number, (title, text) in enumerate(parts, start=1)
        if text.strip()
    ]
    if not rows:
        # Almost always a scan: pages of images with no text layer. Silence would put an
        # empty document in the index and answer questions with nothing.
        raise ValueError(f"No text could be read from {document.name} ({artifact.path})")
    return pl.DataFrame(rows, schema=DOCUMENT_PARTS_SCHEMA)


DOCUMENT_PARTS_SCHEMA = pl.Schema(
    {"document_id": pl.String, "part": pl.Int64, "part_title": pl.String, "text": pl.String}
)


def _fetch(
    document: DocumentConfig, data_dir: Path, client: httpx.Client, now: datetime | None
) -> RawArtifact:
    raw_dir = data_dir / "raw"
    filename = f"{document.name}.{'xml' if document.format == 'jats' else 'pdf'}"
    if document.inbox is None:
        return ingest_file(document.name, str(document.url), filename, raw_dir, client, now)
    handed_over = inbox_dir(data_dir) / document.inbox
    if not handed_over.is_file():
        raise FileNotFoundError(
            f"not in the inbox: download {document.url} into {handed_over.parent} "
            f"as '{document.inbox}'"
        )
    return store_payload(
        document.name, filename, handed_over.read_bytes(), raw_dir, str(document.url), now
    )


def _pdf_parts(path: Path) -> list[tuple[str | None, str]]:
    """One part per page. A PDF has no headings a parser can trust, so parts have no title."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        # Encrypted with permissions rather than with a password (the SCA's standards
        # are): an empty password opens them, and only then can the text be read.
        reader.decrypt("")
    return [(None, page.extract_text() or "") for page in reader.pages]


def _jats_parts(path: Path) -> list[tuple[str | None, str]]:
    """Abstract and top-level sections of an article, each with its heading.

    Only the top level: a nested section's text is inside its parent's, so emitting both
    would index every paragraph twice.
    """
    # Imported here, not at the top: reading a document needs the `rag` extra, and the
    # rest of this module (fetching, and the contract) runs without it.
    from defusedxml import ElementTree

    tree = ElementTree.parse(path)
    # An abstract is a section like any other, but it often carries no heading of its own.
    abstracts = [(_heading(a) or "Abstract", _flat_text(a)) for a in tree.iterfind(".//abstract")]
    sections = [(_heading(s), _flat_text(s)) for s in tree.iterfind(".//body/sec")]
    return abstracts + sections


def _heading(element: Any) -> str | None:
    title = element.find("title")
    return " ".join(title.itertext()).strip() if title is not None else None


def _flat_text(element: Any) -> str:
    """Every paragraph of a section, its subsections included, as one block of text.

    Paragraphs are separated by a blank line, as plain text marks them, so that cutting
    into chunks can prefer a paragraph's end to a sentence's.
    """
    paragraphs = (" ".join(" ".join(p.itertext()).split()) for p in element.iterfind(".//p"))
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
