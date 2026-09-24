"""The corpus' clean layer: each document's text cleared of what is not prose, cut into
chunks, and each chunk filed under the topics it is about.

What is dropped was decided by reading this corpus, not by habit. A PDF's pages carry
running headers and page numbers; articles end in a reference list - titles of other
works, dense in exactly the words a question uses, and saying nothing; contents pages,
indexes and tables are lines of labels and figures. Left in, each becomes a chunk that
outranks the passage that answers. Every rule here is about the shape of text, never
about a subject, and what each document loses is counted in the `documents` table.

A chunk never spans two parts, so it cites one page or one section. The price is a
paragraph broken by a page turn, which becomes two chunks.

The topics are the domain's vocabulary. A chunk is filed under those of its document's
topics whose terms it uses, and under another topic only on stronger evidence - two of
its terms - because the document's topics are a curated prior and one word out of place
("development" in a report on economic development) is not a subject. A chunk that uses
no term at all keeps its document's topics rather than none: a filter that excludes a
passage does so silently. `topics_basis` says which of the two it was.
"""

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import groupby

import pandera.polars as pa
import polars as pl

from mlops_core.adapter import CleanTable
from mlops_core.config import (
    CHUNKS_TABLE,
    DOCUMENTS_TABLE,
    ChunkingConfig,
    CorpusConfig,
    DocumentConfig,
    TopicConfig,
)

logger = logging.getLogger(__name__)

# --- What is not prose ------------------------------------------------------------------
# Thresholds measured on the corpus of 2026-09-24 (17 documents, 1.33 M characters).

# Running headers and footers sit among the first or last lines of a page, and on at
# least half the pages: no sentence repeats like that. Fewer pages are too little
# repetition to tell a header from a coincidence.
EDGE_LINES = 3
FURNITURE_SHARE = 0.5
FURNITURE_MIN_PAGES = 3

# A reference list opens with its heading alone on a line...
REFERENCES_HEADING = re.compile(
    r"^[ \t]*(?:references|bibliography|literature cited)[ \t]*$", re.IGNORECASE | re.MULTILINE
)
# ...and every entry carries a year in brackets, a volume and pages after a year, a DOI
# or a link. The list runs on through the pages that follow while they cite this
# densely: its pages measured 2.2 to 15 per 1,000 characters, body pages at most 1.8. It
# ends at the first page that does not - a catalogue's entries can follow its references.
CITATION = re.compile(
    r"\((?:19|20)\d{2}[a-z]?\)"
    r"|\b(?:19|20)\d{2}[a-z]?, \d+(?:\(\d+\))?, \d+"
    r"|doi\.org/|\bdoi:|\[CrossRef\]|https?://"
)
CITATIONS_PER_1000 = 2.0

# A line that ends in a figure: a table row, a contents line, an index entry, a chart's
# axis. Several in a row are a listing - never prose, which almost never ends two lines
# running on a number - and figures are answered from the tables, not from a PDF's
# scrambled copy of them. The run goes, the prose around it on the same page stays: a
# standard can explain its score and tabulate it on one page.
FIGURE_ENDED = re.compile(r"(?:^|\s)\d[\d.,:\N{EN DASH}-]*(?:,\s?\d[\d.,:\N{EN DASH}-]*)*$")
LISTING_RUN = 4

# What extraction does to words: a form's or a contents page's dotted leaders, a word
# hyphenated at a line end, and a ligature glyph ("ff") split off its word ("co ffee").
DOTTED_LEADER = re.compile(r"(?:[ \t]*\.){4,}")
LINE_END_HYPHEN = re.compile(r"([A-Za-z]*[a-z])-\n([a-z]+)")
LIGATURE_SPLIT = re.compile(r"\b([A-Za-z]{1,6}) (ffi|ffl|ff|fi|fl)([a-z]+)")
WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")

# Less than this, once cleaned, is a caption or a divider, not a passage.
MIN_PART_CHARS = 40

# --- Cutting ----------------------------------------------------------------------------
# A part too long for one chunk is cut at the coarsest boundary that gets each piece under
# the limit: paragraphs, then sentences, then lines, then words. Pieces are then packed
# back together up to the limit, so a chunk ends where a sentence does.
BOUNDARIES = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"(?<=[.!?])\s+(?=[A-Z])"),
    re.compile(r"\n"),
    re.compile(r"\s+"),
)

# --- Topics -----------------------------------------------------------------------------
# Distinct terms of a topic a chunk must use to be filed under it: one for a topic its
# document is filed under, two for any other.
TERMS_IN_DOCUMENT_TOPIC = 1
TERMS_OUTSIDE_DOCUMENT_TOPICS = 2
BASES = ("terms", "document")


@dataclass(frozen=True)
class Part:
    """A page of a PDF or a section of an article, as the raw reader numbered it."""

    number: int
    title: str | None
    text: str


def corpus_tables(
    documents: Sequence[DocumentConfig],
    corpus: CorpusConfig,
    raw: Mapping[str, pl.DataFrame],
    read_at: Mapping[str, datetime],
) -> dict[str, CleanTable]:
    """One row per document, one per chunk. A document never ingested is left out: the
    extract step already said where it has to be put."""
    tagger = TopicTagger(corpus.topics)
    described: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []
    for document in documents:
        frame = raw.get(document.name)
        if frame is None:
            logger.info("%s has not been ingested: the corpus goes without it", document.name)
            continue
        read = [Part(*row) for row in frame.select("part", "part_title", "text").iter_rows()]
        kept = prose(read, paged=document.format == "pdf")
        pieces = [(part, text) for part in kept for text in split(part.text, corpus.chunking)]
        for number, (part, text) in enumerate(pieces, start=1):
            topics, basis = tagger.tag(text, document.topics)
            chunks.append(
                {
                    "chunk_id": f"{document.name}-{number:04d}",
                    "document_id": document.name,
                    "chunk": number,
                    "part": part.number,
                    "part_title": part.title,
                    "text": text,
                    "characters": len(text),
                    "topics": topics,
                    "topics_basis": basis,
                }
            )
        described.append(
            {
                "document_id": document.name,
                "title": document.title,
                "publisher": document.publisher,
                "year": document.year,
                "url": str(document.url),
                "license": document.license,
                "language": document.language,
                "format": document.format,
                "topics": document.topics,
                "retrieved_on": read_at[document.name].date(),
                "parts": len(read),
                "parts_kept": len(kept),
                "characters": sum(len(part.text) for part in read),
                "characters_kept": sum(len(part.text) for part in kept),
                "chunks": len(pieces),
            }
        )
        logger.info(
            "%s: %d of %d parts kept, %d chunks",
            document.name,
            len(kept),
            len(read),
            len(pieces),
        )
    inputs = tuple(str(row["document_id"]) for row in described)
    return {
        DOCUMENTS_TABLE: CleanTable(pl.DataFrame(described, schema=DOCUMENTS_COLUMNS), inputs),
        CHUNKS_TABLE: CleanTable(pl.DataFrame(chunks, schema=CHUNKS_COLUMNS), inputs),
    }


def prose(parts: Sequence[Part], paged: bool) -> list[Part]:
    """The parts with their prose only: page furniture (for pages), reference lists and
    listings removed, words mended, and the parts left with next to nothing dropped."""
    # The document's own words decide which broken words to mend: "e ffect" is joined
    # because the document also writes "effect", and "the first" is not.
    vocabulary = set(WORD.findall("\n".join(part.text for part in parts).lower()))
    if paged:
        parts = _without_furniture(parts)
    kept = []
    for part in _without_references(parts):
        text = _mended(_without_listings(part.text), vocabulary)
        if len(text) >= MIN_PART_CHARS:
            kept.append(replace(part, text=text))
    return kept


def split(text: str, chunking: ChunkingConfig) -> list[str]:
    """Cut a part into chunks of at most `max_chars`, each opening with the last whole
    pieces of the one before, up to `overlap_chars` of them."""
    limit = chunking.max_chars
    windows: list[list[tuple[int, int]]] = []
    window: list[tuple[int, int]] = []
    for piece in _pieces(text, 0, len(text), limit, 0):
        if window and piece[1] - window[0][0] > limit:
            windows.append(window)
            window = _tail(window, chunking.overlap_chars)
            while window and piece[1] - window[0][0] > limit:
                window = window[1:]
        window.append(piece)
    if window:
        windows.append(window)
    chunks = (text[window[0][0] : window[-1][1]].strip() for window in windows)
    return [chunk for chunk in chunks if chunk]


class TopicTagger:
    """Files a text under the topics whose terms it uses."""

    def __init__(self, topics: Mapping[str, TopicConfig]):
        self._patterns = {name: _terms_pattern(topic.terms) for name, topic in topics.items()}

    def tag(self, text: str, declared: Sequence[str]) -> tuple[list[str], str]:
        """The topics, in the vocabulary's order, and whether the terms or the document
        decided them."""
        tagged = [
            name
            for name, pattern in self._patterns.items()
            if len({match.lastgroup for match in pattern.finditer(text)})
            >= (TERMS_IN_DOCUMENT_TOPIC if name in declared else TERMS_OUTSIDE_DOCUMENT_TOPICS)
        ]
        if tagged:
            return tagged, "terms"
        return [name for name in self._patterns if name in declared], "document"


def corpus_contracts(corpus: CorpusConfig) -> dict[str, pa.DataFrameSchema]:
    """The corpus' tables' promises: every topic is one the vocabulary declares, and no
    chunk is longer than the cut allows."""
    topics = pa.Column(
        pl.List(pl.String),
        [
            pa.Check(
                lambda data: data.lazyframe.select(pl.col(data.key).list.len() > 0),
                error="every document and chunk is filed under at least one topic",
            ),
            pa.Check(
                lambda data: data.lazyframe.select(
                    pl.col(data.key).list.eval(pl.element().is_in(list(corpus.topics))).list.all()
                ),
                error="every topic is one corpus.topics declares",
            ),
        ],
    )
    documents = pa.DataFrameSchema(
        name=DOCUMENTS_TABLE,
        strict=True,
        columns={
            "document_id": pa.Column(pl.String, unique=True),
            "title": pa.Column(pl.String),
            "publisher": pa.Column(pl.String),
            "year": pa.Column(pl.Int64, nullable=True),
            "url": pa.Column(pl.String, pa.Check.str_matches(r"^https?://")),
            "license": pa.Column(pl.String),
            "language": pa.Column(pl.String, pa.Check.str_length(2, 2)),
            "format": pa.Column(pl.String, pa.Check.isin(["pdf", "jats"])),
            "topics": topics,
            "retrieved_on": pa.Column(pl.Date),
            "parts": pa.Column(pl.Int64, pa.Check.ge(1)),
            "parts_kept": pa.Column(pl.Int64, pa.Check.ge(0)),
            "characters": pa.Column(pl.Int64, pa.Check.ge(1)),
            "characters_kept": pa.Column(pl.Int64, pa.Check.ge(0)),
            "chunks": pa.Column(pl.Int64, pa.Check.ge(0)),
        },
        checks=[
            pa.Check(
                lambda data: data.lazyframe.select(
                    (pl.col("parts_kept") <= pl.col("parts"))
                    & (pl.col("characters_kept") <= pl.col("characters"))
                ),
                error="cleaning only removes text",
            )
        ],
    )
    chunks = pa.DataFrameSchema(
        name=CHUNKS_TABLE,
        strict=True,
        unique=["document_id", "chunk"],
        columns={
            "chunk_id": pa.Column(pl.String, unique=True),
            "document_id": pa.Column(pl.String),
            "chunk": pa.Column(pl.Int64, pa.Check.ge(1)),
            "part": pa.Column(pl.Int64, pa.Check.ge(1)),
            "part_title": pa.Column(pl.String, nullable=True),
            "text": pa.Column(pl.String, pa.Check.str_length(min_value=1)),
            "characters": pa.Column(pl.Int64, pa.Check.in_range(1, corpus.chunking.max_chars)),
            "topics": topics,
            "topics_basis": pa.Column(pl.String, pa.Check.isin(BASES)),
        },
        checks=[
            pa.Check(
                lambda data: data.lazyframe.select(
                    pl.col("characters") == pl.col("text").str.len_chars()
                ),
                error="characters counts the chunk's text",
            )
        ],
    )
    return {DOCUMENTS_TABLE: documents, CHUNKS_TABLE: chunks}


DOCUMENTS_COLUMNS = pl.Schema(
    {
        "document_id": pl.String(),
        "title": pl.String(),
        "publisher": pl.String(),
        "year": pl.Int64(),
        "url": pl.String(),
        "license": pl.String(),
        "language": pl.String(),
        "format": pl.String(),
        "topics": pl.List(pl.String()),
        "retrieved_on": pl.Date(),
        "parts": pl.Int64(),
        "parts_kept": pl.Int64(),
        "characters": pl.Int64(),
        "characters_kept": pl.Int64(),
        "chunks": pl.Int64(),
    }
)
CHUNKS_COLUMNS = pl.Schema(
    {
        "chunk_id": pl.String(),
        "document_id": pl.String(),
        "chunk": pl.Int64(),
        "part": pl.Int64(),
        "part_title": pl.String(),
        "text": pl.String(),
        "characters": pl.Int64(),
        "topics": pl.List(pl.String()),
        "topics_basis": pl.String(),
    }
)


def _without_furniture(parts: Sequence[Part]) -> list[Part]:
    """Drop the lines that open or close most pages: running headers, footers, page
    numbers, a publisher's download stamp."""
    pages = [part.text.splitlines() for part in parts]
    if len(pages) < FURNITURE_MIN_PAGES:
        return list(parts)
    seen = Counter(
        key for lines in pages for key in {_furniture_key(lines[i]) for i in _edges(lines)}
    )
    furniture = {key for key, count in seen.items() if count >= FURNITURE_SHARE * len(pages)}

    def cleared(lines: Sequence[str]) -> str:
        edges = _edges(lines)
        return "\n".join(
            line
            for i, line in enumerate(lines)
            if i not in edges or _furniture_key(line) not in furniture
        )

    return [replace(part, text=cleared(lines)) for part, lines in zip(parts, pages, strict=True)]


def _edges(lines: Sequence[str]) -> set[int]:
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return set(filled[:EDGE_LINES] + filled[-EDGE_LINES:])


def _furniture_key(line: str) -> str:
    """A line's letters only: the page number and the spacing, which change from page to
    page, do not count. A page number alone has no letters, so they all look alike."""
    return re.sub(r"[^a-z]", "", line.lower())


def _without_references(parts: Sequence[Part]) -> list[Part]:
    """Cut each reference list: from its heading, on through the parts that cite as
    densely as a list does."""
    kept, listing = [], False
    for part in parts:
        if listing and _citations_per_1000(part.text) >= CITATIONS_PER_1000:
            continue
        heading = REFERENCES_HEADING.search(part.text)
        listing = heading is not None
        kept.append(replace(part, text=part.text[: heading.start()]) if heading else part)
    return kept


def _citations_per_1000(text: str) -> float:
    return 1000 * len(CITATION.findall(text)) / max(len(text), 1)


def _without_listings(text: str) -> str:
    """Drop every run of lines that end in a figure. Blank lines neither start nor break
    a run: some extractions put one between every row of a table."""
    lines = text.splitlines()
    filled = [i for i, line in enumerate(lines) if line.strip()]
    dropped: set[int] = set()
    for ended, run in groupby(filled, key=lambda i: bool(FIGURE_ENDED.search(lines[i].strip()))):
        rows = list(run)
        if ended and len(rows) >= LISTING_RUN:
            dropped.update(rows)
    return "\n".join(line for i, line in enumerate(lines) if i not in dropped)


def _mended(text: str, vocabulary: set[str]) -> str:
    """Undo what extraction did to the words, and to the spacing between them."""
    text = DOTTED_LEADER.sub(" ", text)
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    def unbroken(match: re.Match[str]) -> str:
        # "fermen-\ntation" is one word; "wet-\nprocessed" keeps its hyphen, because the
        # document writes "wet-processed" elsewhere on one line.
        hyphenated = f"{match[1]}-{match[2]}"
        return hyphenated if hyphenated.lower() in vocabulary else match[1] + match[2]

    def unsplit(match: re.Match[str]) -> str:
        joined = "".join(match.groups())
        return joined if joined.lower() in vocabulary else match[0]

    return LIGATURE_SPLIT.sub(unsplit, LINE_END_HYPHEN.sub(unbroken, text))


def _pieces(text: str, start: int, end: int, limit: int, level: int) -> list[tuple[int, int]]:
    """Spans of `text` no longer than `limit`, cut at the coarsest boundary that does it."""
    if end - start <= limit:
        return [(start, end)]
    if level == len(BOUNDARIES):  # one "word" longer than a chunk: cut it where it must be
        return [(at, min(at + limit, end)) for at in range(start, end, limit)]
    spans, cursor = [], start
    for match in BOUNDARIES[level].finditer(text, start, end):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, end))
    return [piece for s, e in spans if e > s for piece in _pieces(text, s, e, limit, level + 1)]


def _tail(window: Sequence[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """The last pieces of a chunk that fit in `limit` characters, to open the next one."""
    end = window[-1][1]
    return [span for span in window if end - span[0] <= limit]


def _terms_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    """One pattern for a topic's terms, each a named group so a match says which term it
    was. Longest first, so "roast level" is found as itself and not as "roast"."""
    ordered = sorted(terms, key=len, reverse=True)
    alternatives = "|".join(f"(?P<t{i}>{_inflected(term)})" for i, term in enumerate(ordered))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


def _inflected(term: str) -> str:
    """A term as a pattern for it and its inflections: roast, roasts, roasted, roasting,
    roaster; variety, varieties. Only the last word inflects, and the words of a phrase
    may be joined by a space, a line break, a hyphen or an underscore (a closed
    vocabulary writes "semi_washed", a text "semi-washed")."""
    *head, last = re.split(r"[\s_-]+", term.strip().lower())
    if len(last) > 2 and last.endswith("y") and last[-2] not in "aeiou":
        ending = re.escape(last[:-1]) + "(?:y|ies|ied)"
    elif last.endswith("e"):
        ending = re.escape(last[:-1]) + "(?:e|es|ed|ing|er|ers)"
    else:
        ending = re.escape(last) + "(?:s|es|ed|ing|er|ers)?"
    return r"[\s_-]+".join([*map(re.escape, head), ending])
