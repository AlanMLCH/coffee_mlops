"""The corpus' clean layer: what is kept of a document's text, how it is cut, and how each
chunk is filed.

The texts are written, not recorded: each reproduces one shape seen in the real corpus
(a journal's running header, a catalogue whose varieties follow its references, a
standard that tabulates the score it just explained) in a few lines.
"""

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest
from pandera.errors import SchemaErrors
from pydantic import ValidationError

from domains.coffee.adapter import CoffeeAdapter
from mlops_core.adapter import CleanTable
from mlops_core.config import (
    CHUNKS_TABLE,
    DOCUMENTS_TABLE,
    ChunkingConfig,
    CorpusConfig,
    DocumentConfig,
    TopicConfig,
)
from mlops_core.contracts import check_contract
from mlops_core.data.clean import build_clean
from mlops_core.data.corpus import (
    Part,
    TopicTagger,
    corpus_contracts,
    corpus_tables,
    prose,
    split,
)
from mlops_core.storage import MANIFEST_NAME, read_table

SENTENCE = "Washed coffees are fermented before they are dried on raised beds."

TOPICS = {
    "processing": TopicConfig(
        description="Cherry to green coffee.", terms=["washed", "semi_washed", "drying"]
    ),
    "roasting": TopicConfig(description="The roast.", terms=["roast", "roast level"]),
    "market": TopicConfig(description="Trade.", terms=["export", "import", "price"]),
}
CORPUS = CorpusConfig(topics=TOPICS, chunking=ChunkingConfig(max_chars=200, overlap_chars=80))


def pages(*texts: str) -> list[Part]:
    return [Part(number, None, text) for number, text in enumerate(texts, start=1)]


# --- What is not prose -------------------------------------------------------------------


def test_a_running_header_is_dropped_and_the_same_words_mid_page_are_not() -> None:
    """A journal's header repeats at the top of every page with a new page number. The
    same words in the middle of a page are text - a heading, say - and stay."""
    header = "Beverages 2020, 6, 44 {n} of 9"
    body = [
        [header.format(n=n), *(f"Lot {page}{line} was washed and dried." for line in "abcdefg")]
        for n, page in enumerate("abcd", start=1)
    ]
    body[1][4] = header.format(n=2)

    kept = prose(pages(*("\n".join(lines) for lines in body)), paged=True)

    assert [part.number for part in kept] == [1, 2, 3, 4]
    assert "of 9" not in kept[0].text
    assert kept[1].text.count("of 9") == 1
    assert kept[0].text.startswith("Lot aa was washed")


def test_two_pages_are_too_few_to_call_anything_a_header() -> None:
    kept = prose(pages(f"Chapter one\n{SENTENCE}", f"Chapter one\n{SENTENCE}"), paged=True)

    assert all(part.text.startswith("Chapter one") for part in kept)


def test_a_reference_list_is_cut_from_its_heading_through_the_pages_it_fills() -> None:
    """A catalogue can put its references between its introduction and its entries: the
    list ends at the first page that stops citing, not at the end of the document."""
    entry = "Davis, A. P. (2011). Growing coffee. Botanical Journal, 167(4). https://doi.org/1"
    parts = pages(
        f"{SENTENCE}\nReferences\n{entry}",
        "\n".join([entry] * 6),
        f"Bourbon Mayaguez 139\n{SENTENCE}",
    )

    kept = prose(parts, paged=False)

    assert [part.number for part in kept] == [1, 3]
    assert kept[0].text == SENTENCE
    assert not any("Davis" in part.text for part in kept)


def test_a_table_is_dropped_and_the_explanation_on_its_page_is_kept() -> None:
    """A standard explains its score and tabulates it on one page: the rows go, the page
    stays - and a sentence that happens to end on a number is not a table."""
    explained = "The cupping score is a sum of eight sections."
    rows = "\n".join(f"{total} {58 + total * 0.75:.2f}" for total in range(8, 20))
    after = "Two points are deducted per defective cup.\nIt is rounded to 0.25"

    (kept,) = prose(pages(f"{explained}\n{rows}\n{after}"), paged=False)

    assert kept.text == f"{explained}\n{after}"


def test_words_broken_by_extraction_are_mended_with_the_documents_own_spelling() -> None:
    """A line-end hyphen is a syllable break unless the document writes that compound on
    one line elsewhere; a split ligature is joined only where the document spells the
    word whole, so "the first" is never read as "thefirst"."""
    text = (
        "Beans are fermen-\ntation dried. Wet-processed beans differ from wet-\n"
        "processed ones in name only. Co ffee is roasted; coffee is brewed.\n"
        "Name . . . . . . . the first time"
    )

    (kept,) = prose(pages(text), paged=False)

    assert "fermentation dried" in kept.text
    assert "from wet-processed ones" in kept.text
    assert "Coffee is roasted" in kept.text
    assert "Name the first time" in kept.text


def test_a_part_with_next_to_nothing_left_is_dropped() -> None:
    """ "Not applicable." under an ethics heading is not a passage anyone asks about."""
    parts = [Part(1, "Informed Consent Statement", "Not applicable."), Part(2, "Results", SENTENCE)]

    assert [part.title for part in prose(parts, paged=False)] == ["Results"]


# --- Cutting ------------------------------------------------------------------------------


def test_a_long_part_is_cut_at_sentences_with_whole_sentences_carried_over() -> None:
    sentences = [f"Sentence {n} says what lot {n} tasted like after drying." for n in range(9)]

    chunks = split(" ".join(sentences), CORPUS.chunking)

    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)  # a chunk ends where a sentence does
    for before, after in pairwise(chunks):
        assert before.endswith(after.split(". ")[0] + ".")  # the previous chunk's last one
    assert " ".join(sentences).endswith(chunks[-1])


def test_what_is_carried_over_is_trimmed_to_let_the_next_sentence_fit() -> None:
    """Both short sentences fit in the overlap, but with both the long one would push the
    chunk past its limit: the older one is let go, never the limit."""
    long = "Then " + "a long clause, " * 11 + "and it is done."

    chunks = split(f"Short one. Short two. {long}", CORPUS.chunking)

    assert chunks == ["Short one. Short two.", f"Short two. {long}"]


def test_a_paragraph_end_is_preferred_to_a_sentence_end() -> None:
    """Packed by sentence, the first chunk would take the next paragraph's first sentence."""
    second = f"{SENTENCE} {SENTENCE}"

    chunks = split(f"{SENTENCE}\n\n{second}", ChunkingConfig(max_chars=150, overlap_chars=0))

    assert chunks == [SENTENCE, second]


def test_a_part_that_fits_is_one_chunk_and_a_word_longer_than_a_chunk_is_cut() -> None:
    assert split(SENTENCE, CORPUS.chunking) == [SENTENCE]
    assert [len(chunk) for chunk in split("x" * 450, CORPUS.chunking)] == [200, 200, 50]


# --- Topics -------------------------------------------------------------------------------


def test_a_term_matches_its_inflections_and_never_inside_another_word() -> None:
    tagger = TopicTagger(TOPICS)

    assert tagger.tag("The beans were roasted twice.", ["roasting"]) == (["roasting"], "terms")
    # A closed vocabulary's "semi_washed" is a text's "Semi-washed".
    assert tagger.tag("Semi-washed lots, dried on\nbeds.", ["processing"])[0] == ["processing"]
    # "import" is a market term; "important" is not about imports.
    assert tagger.tag("An important step.", ["market"]) == (["market"], "document")


def test_a_topic_outside_the_documents_needs_two_of_its_terms() -> None:
    """The document's topics are a curated prior: one word out of place is not a subject."""
    tagger = TopicTagger(TOPICS)
    declared = ["processing"]

    assert tagger.tag("Washed lots fetch a higher price.", declared)[0] == ["processing"]
    assert tagger.tag("Washed lots fetch a higher export price.", declared)[0] == [
        "processing",
        "market",
    ]
    # One phrase is one term: "roast level" does not also count as "roast".
    assert tagger.tag("The roast level of washed lots.", declared)[0] == ["processing"]


def test_a_chunk_that_uses_no_term_keeps_its_documents_topics() -> None:
    """Filed under nothing, a search filtered by topic would never find it."""
    assert TopicTagger(TOPICS).tag("Nothing here.", ["market", "processing"]) == (
        ["processing", "market"],
        "document",
    )


# --- Tables -------------------------------------------------------------------------------


def document(name: str, **values: object) -> DocumentConfig:
    fields: dict[str, object] = {
        "name": name,
        "title": "How coffee is processed",
        "publisher": "A publisher",
        "url": "https://publisher.test/paper",
        "license": "CC BY",
        "language": "en",
        "topics": ["processing"],
        "format": "pdf",
    }
    return DocumentConfig.model_validate(fields | values)


def raw_parts(*texts: str) -> pl.DataFrame:
    return pl.DataFrame(
        {"part": list(range(1, len(texts) + 1)), "part_title": [None] * len(texts), "text": texts},
        schema={"part": pl.Int64, "part_title": pl.String, "text": pl.String},
    )


def test_the_tables_meet_their_contracts_and_count_what_was_dropped() -> None:
    read_at = {"paper": datetime(2026, 9, 24, 12, tzinfo=UTC)}
    last = "The export price of washed lots rose again."
    raw = {"paper": raw_parts(" ".join([SENTENCE] * 5), "Not applicable.", last)}

    tables = corpus_tables([document("paper"), document("handed_over_later")], CORPUS, raw, read_at)
    contracts = corpus_contracts(CORPUS)
    documents = check_contract(contracts[DOCUMENTS_TABLE], tables[DOCUMENTS_TABLE].frame)
    chunks = check_contract(contracts[CHUNKS_TABLE], tables[CHUNKS_TABLE].frame)

    (row,) = documents.iter_rows(named=True)  # the one never ingested is left out
    assert (row["parts"], row["parts_kept"], row["chunks"]) == (3, 2, chunks.height)
    assert row["retrieved_on"] == date(2026, 9, 24)
    assert chunks["chunk_id"].to_list()[:2] == ["paper-0001", "paper-0002"]
    assert chunks["part"].to_list()[-1] == 3  # a chunk cites the page it came from
    assert chunks["topics"].to_list()[-1] == ["processing", "market"]
    assert tables[CHUNKS_TABLE].inputs == ("paper",)


def test_a_chunk_an_earlier_one_already_had_is_dropped() -> None:
    """Two catalogues of one publisher open on the same page; four standards end on the
    same address. The first document keeps the text; the chunks are numbered without gaps."""
    read_at = {name: datetime(2026, 9, 24, tzinfo=UTC) for name in ("arabica", "robusta")}
    intro = "About the catalogue. Information is power, and washed coffees need it."
    robusta = "Robusta cherries are dried on the farm, then hulled."
    raw = {
        "arabica": raw_parts(intro, SENTENCE),
        "robusta": raw_parts(intro.replace(" ", "\n", 3), robusta, intro),
    }
    documents = [document("arabica"), document("robusta")]

    tables = corpus_tables(documents, CORPUS, raw, read_at)
    contracts = corpus_contracts(CORPUS)
    described = check_contract(contracts[DOCUMENTS_TABLE], tables[DOCUMENTS_TABLE].frame)
    chunks = check_contract(contracts[CHUNKS_TABLE], tables[CHUNKS_TABLE].frame)

    assert chunks["chunk_id"].to_list() == ["arabica-0001", "arabica-0002", "robusta-0001"]
    assert chunks["text"].to_list()[2] == robusta
    assert described["chunks_repeated"].to_list() == [0, 2]
    with pytest.raises(SchemaErrors, match="no two chunks share their text"):
        check_contract(contracts[CHUNKS_TABLE], pl.concat([chunks, chunks.head(1)]))


def test_a_chunk_filed_under_an_undeclared_topic_breaks_the_contract() -> None:
    read_at = {"paper": datetime(2026, 9, 24, tzinfo=UTC)}
    tables = corpus_tables([document("paper")], CORPUS, {"paper": raw_parts(SENTENCE)}, read_at)
    frame = tables[CHUNKS_TABLE].frame.with_columns(pl.lit(["tasting"]).alias("topics"))

    with pytest.raises(SchemaErrors) as failed:
        check_contract(corpus_contracts(CORPUS)[CHUNKS_TABLE], frame)

    assert set(failed.value.failure_cases["column"]) == {"topics"}


def test_the_clean_layer_builds_the_corpus_tables_beside_the_domains(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    """The fixture publishers serve one-line PDFs and a short article: the PDFs are too
    short to keep, the article's introduction is a chunk - once, since every article is
    that same one - and every ingested document is described; the ones behind a 403 were
    never handed over, so they are absent."""
    data_dir = raw_dir.parent
    documents = coffee_adapter.config.documents

    paths = build_clean(coffee_adapter, data_dir)

    described = read_table(data_dir / "clean" / DOCUMENTS_TABLE)
    chunks = read_table(data_dir / "clean" / CHUNKS_TABLE)
    served = {d.name for d in documents if d.inbox is None}
    assert set(described["document_id"]) == served
    articles = [d.name for d in documents if d.format == "jats"]
    assert chunks["document_id"].to_list() == articles[:1]
    assert chunks["part_title"].to_list() == ["Introduction"]
    repeated = described.filter(pl.col("document_id").is_in(articles[1:]))["chunks_repeated"]
    assert repeated.to_list() == [1] * (len(articles) - 1)
    manifest = json.loads((paths[CHUNKS_TABLE].parent / MANIFEST_NAME).read_text())
    assert set(manifest["inputs"]) == served


def test_a_domain_cannot_build_a_table_the_corpus_owns(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean, contracts = coffee_adapter.clean, dict(coffee_adapter.clean_contracts())

    def with_documents(
        raw: Mapping[str, pl.DataFrame], read_at: Mapping[str, datetime]
    ) -> dict[str, CleanTable]:
        tables = dict(clean(raw, read_at))
        return tables | {DOCUMENTS_TABLE: tables["boroughs"]}

    contracts[DOCUMENTS_TABLE] = contracts["boroughs"]
    monkeypatch.setattr(coffee_adapter, "clean", with_documents)
    monkeypatch.setattr(coffee_adapter, "clean_contracts", lambda: contracts)

    with pytest.raises(ValueError, match="are the corpus' tables"):
        build_clean(coffee_adapter, raw_dir.parent)


# --- Config -------------------------------------------------------------------------------


def test_documents_without_a_corpus_section_are_refused(coffee_adapter: CoffeeAdapter) -> None:
    config = coffee_adapter.config.model_dump()
    del config["corpus"]

    with pytest.raises(ValidationError, match="need a `corpus:` section"):
        type(coffee_adapter.config).model_validate(config)


def test_an_overlap_as_long_as_a_chunk_is_refused() -> None:
    """Every chunk would repeat the one before it and the cut would never advance."""
    with pytest.raises(ValidationError, match="overlap_chars must be shorter"):
        ChunkingConfig(max_chars=200, overlap_chars=200)
