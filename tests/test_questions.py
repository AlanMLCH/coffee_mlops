"""The retrieval question set: how drafts are made, kept and reviewed.

The local model is replayed from a recorded Ollama exchange (tests/fixtures/ollama/);
the drafting logic is exercised with a scripted drafter, because what matters there is
which passages are asked about and which excerpts can be labels, not what a model says.
"""

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import domains.coffee
from domains.coffee.adapter import CoffeeAdapter
from mlops_core import cli
from mlops_core.adapter import domain_dir
from mlops_core.config import CHUNKS_TABLE, DOCUMENTS_TABLE, TopicConfig
from mlops_core.data.corpus import CHUNKS_COLUMNS
from mlops_core.rag import llm
from mlops_core.rag.llm import LocalModel, ollama_client
from mlops_core.rag.questions import (
    DRAFTING_MODEL,
    MAX_EXCERPT_CHARS,
    OPTIONS,
    PROMPT_VERSION,
    Draft,
    Judgment,
    Question,
    contains,
    draft_questions,
    load_questions,
    questions_path,
    reviewed,
    save_questions,
    tally,
)
from mlops_core.storage import read_table, write_table

OLLAMA = Path(__file__).parent / "fixtures" / "ollama"
TODAY = date(2026, 9, 24)
TOPICS = {
    "roasting": TopicConfig(description="The roast.", terms=["roast"]),
    "brewing": TopicConfig(description="Extraction.", terms=["brew"]),
}
# Every chunk ends with the same filler, as a catalogue repeats its boilerplate.
FILLER = " ".join(["The beans were sorted, weighed and set aside for the panel."] * 10)
# What the recorded drafter quoted, so the one chunk that holds it can be drafted from.
RECORDED_EXCERPT = "Intensity does not imply quality or desirability."


def text_of(document: str, part: int) -> str:
    """A passage whose first sentence is its own."""
    return f"Lot {document}{part} was roasted until the first crack. {FILLER}"


def chunk(document: str, part: int, topics: list[str], **values: Any) -> dict[str, Any]:
    text = values.pop("text", text_of(document, part))
    return {
        "chunk_id": f"{document}-{part:04d}",
        "document_id": document,
        "chunk": part,
        "part": part,
        "part_title": None,
        "text": text,
        "characters": len(text),
        "topics": topics,
        "topics_basis": "terms",
    } | values


def chunks(*rows: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=CHUNKS_COLUMNS)


def documents(*names: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "document_id": list(names),
            "title": [f"On {n}" for n in names],
            "publisher": ["P"] * len(names),
        }
    )


def first_sentence(prompt: str) -> str:
    passage = prompt.split("<passage>\n", 1)[1]
    return passage.split(". ", 1)[0] + "."


def drafter(
    excerpt: Callable[[str], str] = first_sentence, declined: frozenset[str] = frozenset()
) -> tuple[Callable[[str], Draft], list[str]]:
    """A scripted drafter: it quotes what `excerpt` picks, declines passages holding any
    of `declined`, and keeps every prompt it was given."""
    prompts: list[str] = []

    def ask(prompt: str) -> Draft:
        prompts.append(prompt)
        return Draft(
            usable=not any(word in prompt for word in declined),
            evidence=excerpt(prompt),
            question=f"Question {len(prompts)}?",
            answer="An answer.",
        )

    return ask, prompts


def question(qid: str, document: str, part: int, status: str = "draft") -> Question:
    excerpt = text_of(document, part).split(". ", 1)[0] + "."
    return Question(
        id=qid,
        topic=qid.rsplit("-", 1)[0],
        question="How does a roast develop?",
        answer="It browns.",
        relevant=[Judgment(document_id=document, part=part, excerpt=excerpt, grade=2)],
        status=status,
        drafted_by="model@abc",
        prompt=PROMPT_VERSION,
        drafted_on=TODAY,
        source_chunk=f"{document}-{part:04d}",
    )


# --- Drafting ------------------------------------------------------------------------------


def test_a_topics_questions_are_dealt_from_its_documents_in_turn_largest_first() -> None:
    """No single document writes a topic's questions, and a round cut short favours the
    document that holds most of the topic."""
    corpus = chunks(
        chunk("small", 1, ["roasting"]), *(chunk("big", part, ["roasting"]) for part in (1, 2, 3))
    )
    ask, _ = drafter()

    drafted = list(
        draft_questions(corpus, documents("big", "small"), TOPICS, [], 3, ask, "m@1", TODAY)
    )

    assert [q.id for q in drafted] == ["roasting-01", "roasting-02", "roasting-03"]
    assert [q.source.document_id for q in drafted] == ["big", "small", "big"]
    assert drafted[0].source.excerpt.startswith("Lot big")
    assert all(q.status == "draft" and q.prompt == PROMPT_VERSION for q in drafted)


def test_only_substantial_passages_filed_under_the_topic_by_their_terms_are_asked_about() -> None:
    corpus = chunks(
        chunk("a", 1, ["roasting"], topics_basis="document"),  # filed by its document
        chunk("a", 2, ["roasting"], text="Too short."),
        chunk("a", 3, ["brewing"]),  # another topic's
        chunk("a", 4, ["roasting", "brewing"]),
    )
    ask, prompts = drafter()

    drafted = list(draft_questions(corpus, documents("a"), TOPICS, [], 5, ask, "m@1", TODAY))

    assert [(q.topic, q.source.part) for q in drafted] == [("roasting", 4), ("brewing", 3)]
    assert "Topic: roasting - The roast." in prompts[0]
    assert 'Document: "On a", P' in prompts[0]


def test_a_topic_is_topped_up_without_reusing_a_passage_or_an_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A chunk holding an earlier question's excerpt is not asked about again - a
    rejected one's included, since that passage was tried - and no id is given twice. A
    topic that runs out of passages says so."""
    corpus = chunks(*(chunk("a", part, ["roasting"]) for part in (1, 2, 3)))
    existing = [
        question("roasting-01", "a", 1, "accepted"),
        question("roasting-02", "a", 2, "rejected"),
    ]
    ask, _ = drafter()

    drafted = list(draft_questions(corpus, documents("a"), TOPICS, existing, 3, ask, "m@1", TODAY))

    assert [(q.id, q.source.part) for q in drafted] == [("roasting-03", 3)]
    assert "roasting: 1 questions short" in caplog.text


def test_a_passage_the_drafter_declines_is_passed_over() -> None:
    corpus = chunks(
        chunk("a", 1, ["roasting"], text="References. " + FILLER), chunk("a", 2, ["roasting"])
    )
    ask, prompts = drafter(declined=frozenset({"References"}))

    drafted = list(draft_questions(corpus, documents("a"), TOPICS, [], 1, ask, "m@1", TODAY))

    assert len(prompts) == 2
    assert [q.source.part for q in drafted] == [2]


@pytest.mark.parametrize(
    ("excerpt", "flaw"),
    [
        (lambda prompt: "Lots were roasted lightly.", "does not contain"),  # a paraphrase
        (lambda prompt: first_sentence(prompt) + " " + FILLER, "a sentence or two"),
        # Boilerplate every chunk carries: any search would find it.
        (lambda prompt: FILLER.split(". ")[0] + ".", "found on 1 other pages"),
    ],
)
def test_an_excerpt_that_cannot_be_a_label_is_not_one(
    caplog: pytest.LogCaptureFixture, excerpt: Callable[[str], str], flaw: str
) -> None:
    """A label must quote the corpus, be a sentence or two, and point at one passage."""
    caplog.set_level(logging.INFO)
    corpus = chunks(chunk("a", 1, ["roasting"]), chunk("b", 1, ["brewing"]))
    ask, _ = drafter(excerpt=excerpt)

    drafted = list(draft_questions(corpus, documents("a", "b"), TOPICS, [], 1, ask, "m@1", TODAY))

    assert drafted == []
    assert flaw in caplog.text


def test_an_excerpt_is_found_whatever_the_line_breaks_and_case() -> None:
    assert contains("Roasting turns\ngreen  beans brown.", "roasting turns green beans brown.")
    assert not contains("Roasting turns green beans brown.", "Roasting turns beans brown.")


# --- Keeping and reviewing -----------------------------------------------------------------


def test_the_set_is_kept_one_line_per_question_in_id_order(tmp_path: Path) -> None:
    path = tmp_path / "evals" / "questions.jsonl"
    save_questions(path, [question("roasting-02", "a", 2), question("brewing-01", "a", 1)])

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["brewing-01", "roasting-02"]
    assert "reviewed_on" not in json.loads(lines[0])  # a draft has not been reviewed
    assert load_questions(path, TOPICS) == [
        question("brewing-01", "a", 1),
        question("roasting-02", "a", 2),
    ]
    assert load_questions(tmp_path / "none.jsonl", TOPICS) == []


@pytest.mark.parametrize(
    ("questions", "error"),
    [
        ([question("tasting-01", "a", 1)], "topics corpus.topics lacks"),
        ([question("roasting-01", "a", 1), question("roasting-01", "a", 2)], "repeated"),
    ],
)
def test_a_set_that_contradicts_the_corpus_is_refused(
    tmp_path: Path, questions: list[Question], error: str
) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_questions(path, TOPICS)


def test_the_first_label_must_be_a_passage_that_answers_the_question() -> None:
    fields = question("roasting-01", "a", 1).model_dump()
    fields["relevant"][0]["grade"] = 1

    with pytest.raises(ValidationError, match="graded 2"):
        Question.model_validate(fields)


def test_a_review_records_whether_the_draft_was_taken_as_written() -> None:
    draft = question("roasting-01", "a", 1)

    assert reviewed(draft, "accept", TODAY).status == "accepted"
    assert reviewed(draft, "accept", TODAY, wording=f" {draft.question} ").status == "accepted"
    edited = reviewed(draft, "accept", TODAY, wording="Why does a roast brown?", answer="")
    assert (edited.status, edited.question, edited.answer) == (
        "edited",
        "Why does a roast brown?",
        draft.answer,
    )
    assert reviewed(draft, "reject", TODAY).status == "rejected"
    assert reviewed(draft, "reject", TODAY).reviewed_on == TODAY
    assert tally([draft, edited]) == {"roasting": {"draft": 1, "edited": 1}}


# --- The local model -----------------------------------------------------------------------


def recorded_ollama(seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    replies = {
        "/api/tags": json.loads((OLLAMA / "tags.json").read_text(encoding="utf-8")),
        "/api/chat": json.loads((OLLAMA / "chat.json").read_text(encoding="utf-8")),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=replies[request.url.path])

    return httpx.MockTransport(handler)


def test_a_reply_is_constrained_to_the_schema_and_read_back_through_it() -> None:
    seen: list[httpx.Request] = []
    with ollama_client("http://ollama.test", recorded_ollama(seen)) as client:
        model = LocalModel(client, DRAFTING_MODEL, OPTIONS)
        digest = model.digest()
        draft = model.ask("Write a question.", Draft)

    assert digest == "2a654d98e6fb"
    assert draft.usable and draft.question.endswith("?")
    assert draft.evidence == RECORDED_EXCERPT
    body = json.loads(seen[-1].content)
    assert body["format"] == Draft.model_json_schema()
    assert list(body["format"]["properties"]) == ["usable", "evidence", "question", "answer"]
    assert (body["think"], body["stream"], body["options"]) == (False, False, OPTIONS)


def test_a_model_ollama_does_not_have_is_named_with_how_to_get_it() -> None:
    with (
        ollama_client("http://ollama.test", recorded_ollama()) as client,
        pytest.raises(LookupError, match="ollama pull nope:1b"),
    ):
        LocalModel(client, "nope:1b", OPTIONS).digest()


def test_ollama_not_running_is_said_plainly() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with (
        ollama_client("http://ollama.test", httpx.MockTransport(refuse)) as client,
        pytest.raises(ConnectionError, match="start it"),
    ):
        LocalModel(client, DRAFTING_MODEL, OPTIONS).digest()


# --- The commands --------------------------------------------------------------------------


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A data dir with a clean corpus of two documents, and a domain dir in a temp
    folder: a test never writes the real question set. Only the catalogue's page holds
    what the recorded drafter quotes."""
    clean = tmp_path / "data" / "coffee" / "clean"
    catalogue = f"{text_of('wcr_arabica_catalog', 7)} {RECORDED_EXCERPT}"
    write_table(
        chunks(
            chunk("wcr_arabica_catalog", 7, ["varieties"], text=catalogue),
            chunk("pmc_roast_aroma", 3, ["roasting"], part_title="Results"),
        ),
        clean / CHUNKS_TABLE,
        {},
    )
    write_table(
        pl.DataFrame(
            {
                "document_id": ["wcr_arabica_catalog", "pmc_roast_aroma"],
                "title": ["Arabica Coffee Varieties", "Roast level and aroma"],
                "publisher": ["World Coffee Research", "Molecules"],
            }
        ),
        clean / DOCUMENTS_TABLE,
        {},
    )
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "domain_dir", lambda domain: tmp_path / "domain")
    return tmp_path


def served(transport: httpx.BaseTransport) -> Callable[..., Any]:
    @contextmanager
    def client(url: str) -> Iterator[httpx.Client]:
        with ollama_client(url, transport) as c:
            yield c

    return client


def coffee_topics() -> dict[str, TopicConfig]:
    corpus = domains.coffee.adapter().config.corpus
    assert corpus is not None
    return corpus.topics


def test_draft_saves_each_question_as_it_is_written(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded drafter quotes the same sentence for every passage: the page that
    holds it gets a question, the one that does not is passed over."""
    monkeypatch.setattr(llm, "ollama_client", served(recorded_ollama()))

    result = CliRunner().invoke(cli.app, ["rag", "draft", "--domain", "coffee", "--per-topic", "1"])

    assert result.exit_code == 0, result.output
    saved = load_questions(questions_path(workspace / "domain"), coffee_topics())
    assert [q.id for q in saved] == ["varieties-01"]
    assert saved[0].drafted_by == "qwen3.5:4b@2a654d98e6fb"
    assert saved[0].source.excerpt == RECORDED_EXCERPT
    assert "varieties: 1 draft" in result.output


def test_draft_stops_before_drafting_when_the_model_is_not_there(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "ollama_client", served(recorded_ollama()))

    result = CliRunner().invoke(cli.app, ["rag", "draft", "--drafter", "nope:1b"])

    assert result.exit_code == 1
    assert "ollama pull nope:1b" in result.output
    assert not questions_path(workspace / "domain").exists()


def test_review_saves_every_decision_and_leaves_skipped_drafts_waiting(workspace: Path) -> None:
    path = questions_path(workspace / "domain")
    drafts = [
        question("roasting-01", "pmc_roast_aroma", 3),
        question("roasting-02", "pmc_roast_aroma", 3),
        question("roasting-03", "pmc_roast_aroma", 3),
        # Cut again since it was drafted: the chunk that holds its excerpt is shown.
        question("varieties-01", "wcr_arabica_catalog", 7).model_copy(
            update={"source_chunk": "gone-0001"}
        ),
    ]
    save_questions(path, drafts)

    result = CliRunner().invoke(
        cli.app, ["rag", "review"], input="a\ne\nWhy does a roast turn brown?\n\nr\ns\n"
    )

    assert result.exit_code == 0, result.output
    assert "section 'Results'" in result.output
    assert "page 7" in result.output
    assert f"Excerpt (the label): {drafts[3].source.excerpt}" in result.output
    statuses = {q.id: q.status for q in load_questions(path, coffee_topics())}
    assert statuses == {
        "roasting-01": "accepted",
        "roasting-02": "edited",
        "roasting-03": "rejected",
        "varieties-01": "draft",
    }

    quit_at_once = CliRunner().invoke(cli.app, ["rag", "review"], input="q\n")
    assert quit_at_once.exit_code == 0
    assert {q.id: q.status for q in load_questions(path, coffee_topics())} == statuses


def test_evaluate_scores_the_set_and_writes_a_table_sql_can_read(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)  # MLflow writes local artifacts under the working dir
    save_questions(
        questions_path(workspace / "domain"), [question("roasting-01", "pmc_roast_aroma", 3)]
    )

    result = CliRunner().invoke(cli.app, ["rag", "evaluate"])

    assert result.exit_code == 0, result.output
    assert "recall_at_10: 1.000" in result.output  # two chunks: it is in the top ten
    evaluated = workspace / "data" / "coffee" / "evaluations" / "retrieval_bm25"
    assert read_table(evaluated)["question_id"].to_list() == ["roasting-01"]


def test_a_domain_without_a_corpus_has_no_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    config = domains.coffee.adapter().config
    bare = CoffeeAdapter(config.model_copy(update={"documents": [], "corpus": None}))
    monkeypatch.setattr(cli, "load_adapter", lambda domain: bare)

    result = CliRunner().invoke(cli.app, ["rag", "review"])

    assert result.exit_code == 1
    assert "has no corpus" in result.output


def test_the_committed_question_set_speaks_the_corpus_vocabulary() -> None:
    """The real set: every topic declared, every source a listed document, every label a
    sentence or two."""
    config = domains.coffee.adapter().config
    questions = load_questions(questions_path(domain_dir("coffee")), coffee_topics())

    listed = {document.name for document in config.documents}
    assert {q.source.document_id for q in questions} <= listed
    assert all(len(q.source.excerpt) <= MAX_EXCERPT_CHARS for q in questions)
