"""The retrieval evaluation set: questions a person has checked, each with the passages
that answer it.

A local model drafts; a person decides. Drafting is the cheap half - a question from a
passage in a couple of seconds - and reviewing is what makes the set evidence: a
question nobody checked measures the drafter, not the search. Every draft is kept with
what its reviewer did to it (accepted, edited, rejected), so the set also records how
far a small model's drafts can be trusted.

A label is an excerpt: the sentence of the corpus, word for word, that answers the
question, with the document and page or section it is on. A retrieved chunk is relevant
when it contains the excerpt - whatever the cut. Labelling chunks would break with every
change to the cut, which is one of the settings this set exists to tune; labelling a
page or section would be too coarse, since an article's section can run to thirty
chunks and any of them would count as a hit. The drafter copies the excerpt and the
copy is checked against the passage, so a label never quotes what the corpus does not
say. The source excerpt is the first label, graded 2 ("answers it"); passages judged
later, when several searches pool their results, are appended with their own grade, 0
included, because "judged not relevant" is not "never judged".

A question drafted from a passage tends to borrow its words, which flatters lexical
search. The prompt asks for the drafter's own words and a reviewer can reword the rest;
what remains is a known bias of the set, stated rather than assumed away.

The set is data the domain owns, versioned beside its code: one JSON line per question,
in `<domain package>/evals/retrieval_questions.jsonl`.
"""

import hashlib
import json
import logging
import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlops_core.config import TopicConfig

logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path("evals") / "retrieval_questions.jsonl"

Status = Literal["draft", "accepted", "edited", "rejected"]

# Chosen by trying both installed models on the same five passages (2026-09-24):
# qwen3.5:4b wrote self-contained questions with answers in its own words; granite4.2:3b
# wrote "the passage" into its questions and a second question where the answer belonged.
DRAFTING_MODEL = "qwen3.5:4b"
# Some variety between drafts, and the same drafts again for the same model and seed.
OPTIONS = {"temperature": 0.7, "seed": 7}
# Shorter than this, a chunk is a caption or the tail of a page, not a passage to ask about.
MIN_PASSAGE_CHARS = 600
# An excerpt is a sentence or two. Longer, it is the passage copied whole, and a later
# cut could split it between two chunks so that neither contains it.
MAX_EXCERPT_CHARS = 400

PROMPT = """You are writing test questions for a search engine over a library of documents.

Read the passage below. Write ONE question that a curious professional in this field
might ask, and that this passage answers well.

Rules:
- The question must make sense on its own to someone who has never seen the passage:
  never write "the passage", "the text", "the study", "the document" or "the authors",
  and never say "this" about something only the passage names.
- Ask one thing.
- Use your own words. Do not copy distinctive phrases from the passage.
- Ask about an explanation, a cause, a method, a definition or a comparison. If the
  passage mostly reports figures, ask what explains them or what they mean, not what
  they are.
- The answer is one to three sentences in your own words, saying only what the
  passage says.
- The evidence is the single sentence of the passage that best answers the question,
  copied exactly, word for word. One sentence; two only if the answer needs both.
- If the passage is not prose that answers a real question - a list of references, a
  table of contents, a form with blank fields, a fragment - set usable to false.

Topic: {topic} - {description}
Document: "{title}", {publisher}

<passage>
{text}
</passage>
"""


class Draft(BaseModel):
    """What the drafter is asked for. `usable` lets it decline a passage that slipped
    past the cleaning (a reference list, a form) instead of inventing a question.

    The order of the fields is the order the model writes them in, and it matters: with
    the evidence picked first and the question asked about it second, 11 excerpts of 11
    came out verbatim on the same passages where evidence-last managed 10, with no more
    of the passage's words lifted into the question (2026-09-24).
    """

    usable: bool
    evidence: str
    question: str
    answer: str


# What a draft was made with, in eight characters: the prompt, the reply's schema and
# the sampling options - any of them changes what the drafter writes.
PROMPT_VERSION = hashlib.sha256(
    (
        PROMPT
        + json.dumps(Draft.model_json_schema(), sort_keys=True)
        + json.dumps(OPTIONS, sort_keys=True)
    ).encode()
).hexdigest()[:8]


class Judgment(BaseModel):
    """How well one passage answers a question: 2 answers it, 1 bears on it, 0 was
    judged and does not. The excerpt identifies the passage; the document and part say
    where it is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    part: int = Field(ge=1)
    excerpt: str = Field(min_length=1, max_length=MAX_EXCERPT_CHARS)
    grade: Literal[0, 1, 2]


class Question(BaseModel):
    """One question of the set, and everything needed to trust it or not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str  # "<topic>-<nn>", never reused, not even after a rejection
    topic: str
    question: str = Field(min_length=1)
    # The reference answer, from the source passage only: for the reviewer now, and for
    # judging generated answers later.
    answer: str = Field(min_length=1)
    relevant: list[Judgment] = Field(min_length=1)  # the source passage first
    status: Status
    drafted_by: str  # model@digest
    prompt: str  # PROMPT_VERSION
    drafted_on: date
    # The chunk it was drafted from, to show the reviewer. It goes stale when the corpus
    # is cut differently; the labels do not.
    source_chunk: str
    reviewed_on: date | None = None

    @model_validator(mode="after")
    def _the_source_answers_it(self) -> Self:
        if self.relevant[0].grade != 2:
            raise ValueError(f"{self.id}: the first label is the source passage, graded 2")
        return self

    @property
    def source(self) -> Judgment:
        return self.relevant[0]


def contains(text: str, excerpt: str) -> bool:
    """Whether `text` holds `excerpt`, whatever the line breaks and spacing: a PDF breaks
    a sentence where its column ends, and a quote of it does not."""
    return _flat(excerpt) in _flat(text)


def questions_path(domain_dir: Path) -> Path:
    return domain_dir / QUESTIONS_FILE


def load_questions(path: Path, topics: Mapping[str, TopicConfig]) -> list[Question]:
    """The set, validated: every question well formed, filed under a declared topic, and
    named once. No file yet is an empty set."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [Question.model_validate_json(line) for line in lines if line.strip()]
    unknown = sorted({q.topic for q in questions} - set(topics))
    if unknown:
        raise ValueError(f"Questions filed under topics corpus.topics lacks: {unknown}")
    ids = [q.id for q in questions]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"Question ids must be unique; repeated: {repeated}")
    return questions


def save_questions(path: Path, questions: Sequence[Question]) -> None:
    """One line per question, ordered by id, LF endings: a review is a readable diff.
    Written to a temporary file and moved into place, so an interrupted save never
    leaves half a set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (
        json.dumps(q.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
        for q in sorted(questions, key=lambda q: q.id)
    )
    partial = path.with_suffix(".partial")
    partial.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
    partial.replace(path)


def draft_questions(
    chunks: pl.DataFrame,
    documents: pl.DataFrame,
    topics: Mapping[str, TopicConfig],
    existing: Sequence[Question],
    per_topic: int,
    ask: Callable[[str], Draft],
    drafted_by: str,
    today: date,
) -> Iterator[Question]:
    """Top every topic up to `per_topic` questions nobody rejected, yielding each draft
    as it is written so the caller can save as it goes.

    No chunk is asked about twice: not one that holds the excerpt of an earlier question,
    a rejected one included, since that passage was already tried. A topic's passages are
    dealt from its documents in turn, so no single document writes its questions.
    """
    excerpts = [q.source.excerpt for q in existing]
    titles = {row["document_id"]: row for row in documents.iter_rows(named=True)}
    places = [
        (chunk["document_id"], chunk["part"], _flat(chunk["text"]))
        for chunk in chunks.iter_rows(named=True)
    ]
    for topic, vocabulary in topics.items():
        mine = [q for q in existing if q.topic == topic]
        wanted = per_topic - sum(q.status != "rejected" for q in mine)
        number = max((int(q.id.rsplit("-", 1)[1]) for q in mine), default=0)
        for row in _passages(chunks, topic):
            if wanted <= 0:
                break
            if any(contains(row["text"], excerpt) for excerpt in excerpts):
                continue
            document = titles[row["document_id"]]
            draft = ask(
                PROMPT.format(
                    topic=topic,
                    description=vocabulary.description,
                    title=document["title"],
                    publisher=document["publisher"],
                    text=row["text"],
                )
            )
            if not draft.usable:
                logger.info("%s: the drafter declined %s", topic, row["chunk_id"])
                continue
            excerpt = " ".join(draft.evidence.split())
            flaw = _flaw(excerpt, row, places)
            if flaw:
                logger.info("%s: %s %s", topic, row["chunk_id"], flaw)
                continue
            number += 1
            wanted -= 1
            excerpts.append(excerpt)
            yield Question(
                id=f"{topic}-{number:02d}",
                topic=topic,
                question=draft.question.strip(),
                answer=draft.answer.strip(),
                relevant=[
                    Judgment(
                        document_id=row["document_id"], part=row["part"], excerpt=excerpt, grade=2
                    )
                ],
                status="draft",
                drafted_by=drafted_by,
                prompt=PROMPT_VERSION,
                drafted_on=today,
                source_chunk=row["chunk_id"],
            )
        if wanted > 0:
            logger.warning("%s: %d questions short - the topic ran out of passages", topic, wanted)


def reviewed(
    question: Question,
    verdict: Literal["accept", "reject"],
    today: date,
    wording: str | None = None,
    answer: str | None = None,
) -> Question:
    """The question as its reviewer left it. Accepting with changed wording is an edit,
    and is recorded as one: how often the drafts needed fixing is part of the result."""
    if verdict == "reject":
        return question.model_copy(update={"status": "rejected", "reviewed_on": today})
    proposed = {"question": (wording or "").strip(), "answer": (answer or "").strip()}
    changed = {f: text for f, text in proposed.items() if text and text != getattr(question, f)}
    status: Status = "edited" if changed else "accepted"
    return question.model_copy(update={**changed, "status": status, "reviewed_on": today})


def tally(questions: Sequence[Question]) -> dict[str, Counter[str]]:
    """Per topic, how many questions stand in each status."""
    counts: dict[str, Counter[str]] = {}
    for question in questions:
        counts.setdefault(question.topic, Counter())[question.status] += 1
    return counts


def _flat(text: str) -> str:
    return " ".join(text.split()).casefold()


def _flaw(excerpt: str, row: Mapping[str, Any], places: Sequence[tuple[str, int, str]]) -> str:
    """Why an excerpt cannot be a label; empty when it can."""
    if not excerpt or len(excerpt) > MAX_EXCERPT_CHARS:
        return "drew no excerpt of a sentence or two"
    if not contains(row["text"], excerpt):
        # A model that paraphrases where it was told to copy: the label would point at
        # words the corpus does not have.
        return "drew an excerpt it does not contain"
    flat = _flat(excerpt)
    found = {(document, part) for document, part, text in places if flat in text}
    if found - {(row["document_id"], row["part"])}:
        # A stock phrase ("see Table 2.") is in dozens of chunks, and a catalogue repeats
        # the same line on page after page: every search would find a match.
        return f"drew an excerpt found on {len(found) - 1} other pages or sections"
    return ""


def _passages(chunks: pl.DataFrame, topic: str) -> Iterator[dict[str, Any]]:
    """The topic's passages in drafting order: filed under it by their own terms, long
    enough to ask about, and dealt from each document in turn - the document with most
    of them first, so a round cut short still favours where the topic is - shuffled
    within each document with a fixed seed, so the same corpus gives the same order."""
    pool = chunks.filter(
        pl.col("topics").list.contains(topic),
        pl.col("topics_basis") == "terms",
        pl.col("characters") >= MIN_PASSAGE_CHARS,
    )
    rng = random.Random(f"{OPTIONS['seed']}:{topic}")
    by_document = pool.sort("document_id", "chunk").partition_by("document_id", maintain_order=True)
    decks = sorted(
        (list(frame.iter_rows(named=True)) for frame in by_document), key=len, reverse=True
    )
    for deck in decks:
        rng.shuffle(deck)
    for turn in range(max((len(deck) for deck in decks), default=0)):
        for deck in decks:
            if turn < len(deck):
                yield deck[turn]
