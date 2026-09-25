"""Judging retrieval: the metrics, the keyword baseline, and an evaluation run.

Metrics are checked against values worked out by hand; BM25 against the properties that
make it BM25 (rare words weigh more, long texts are discounted, a word counts once in a
query); the run against what it must record.
"""

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path

import mlflow
import polars as pl
import pytest
from mlflow.tracking import MlflowClient

import domains.coffee
from mlops_core.config import ChunkingConfig
from mlops_core.data.corpus import CHUNKS_COLUMNS
from mlops_core.rag.evaluate import evaluate_retrieval, experiment_name, judged_against
from mlops_core.rag.lexical import Bm25, query_vector, term_id, terms
from mlops_core.rag.metrics import grades, ranking_metrics
from mlops_core.rag.questions import PROMPT_VERSION, Judgment, Question
from mlops_core.storage import read_table


def label(excerpt: str, grade: int = 2, document: str = "a") -> Judgment:
    return Judgment(document_id=document, part=1, excerpt=excerpt, grade=grade)


# --- Metrics -------------------------------------------------------------------------------


def test_a_label_is_credited_once_to_the_first_chunk_of_its_document_that_holds_it() -> None:
    """Overlapping neighbours both hold an excerpt; returning both is one find, not two.
    The same words in another document are not the labelled passage."""
    labels = [label("Washed coffee is fermented.")]
    retrieved = [
        ("b", "Washed coffee is fermented."),  # another document
        ("a", "Before drying, washed  coffee\nis fermented. Then it dries."),
        ("a", "washed coffee is fermented. It dries on beds."),  # the overlapping neighbour
    ]

    assert grades(retrieved, labels) == [0, 2, 0]


def test_a_single_label_found_second() -> None:
    metrics = ranking_metrics([0, 2, 0], [label("x")])

    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["average_precision"] == 0.5
    assert (metrics["recall_at_1"], metrics["recall_at_3"]) == (0.0, 1.0)
    assert metrics["ndcg_at_3"] == pytest.approx(1 / math.log2(3))


def test_graded_labels_part_the_metrics() -> None:
    """Two labels, graded 2 and 1, found at ranks 3 and 1."""
    metrics = ranking_metrics([1, 0, 2], [label("x", 2), label("y", 1)])

    assert metrics["reciprocal_rank"] == 1.0
    assert metrics["average_precision"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert metrics["recall_at_1"] == 0.5
    ideal = 3 + 1 / math.log2(3)
    assert metrics["ndcg_at_3"] == pytest.approx((1 + 3 / math.log2(4)) / ideal)


def test_nothing_to_find_scores_zero_rather_than_dividing_by_it() -> None:
    metrics = ranking_metrics([0, 0], [label("x", 0)])

    assert set(metrics.values()) == {0.0}


# --- BM25 ----------------------------------------------------------------------------------


def test_terms_drop_stop_words_and_share_a_stem() -> None:
    assert terms("The roasting of washed coffee") == terms("roasted washing coffees")
    assert "the" not in terms("The roast")


def test_a_rare_word_outweighs_a_common_one() -> None:
    index = Bm25(["coffee roast", "coffee acidity", "coffee body", "coffee aroma"])

    assert index.idf("roast") > index.idf("coffe") > 0
    assert index.search("coffee roasting", 4)[0] == 0


def test_a_long_text_is_discounted_and_a_query_word_counts_once() -> None:
    index = Bm25(["crack", "crack " + "filler " * 20, "unrelated words here"])

    scores = index.scores("crack")
    assert scores[0] > scores[1] > 0
    assert list(index.scores("crack crack crack")) == list(scores)
    # A text that shares no word with the query is never returned, however short the list.
    assert index.search("crack", 3) == [0, 1]


def test_the_sparse_encoding_scores_as_bm25_once_the_index_applies_idf() -> None:
    """What Qdrant computes - document weights times the query's IDF - is BM25."""
    texts = ["dark roast bitter", "light roast", "cupping form"]
    index = Bm25(texts)
    ids, ones = query_vector("dark roast roast")
    by_id = {term_id(term): term for term in ("dark", "roast")}

    assert ones == [1.0, 1.0]  # distinct terms, each once
    for position, text in enumerate(texts):
        doc_ids, weights = index.document_vector(position)
        weighted = dict(zip(doc_ids, weights, strict=True))
        expected = sum(index.idf(by_id[i]) * weighted[i] for i in ids if i in weighted)
        assert index.scores("dark roast")[position] == pytest.approx(expected), text


# --- A run ---------------------------------------------------------------------------------


def question(qid: str, excerpt: str, status: str = "draft") -> Question:
    return Question(
        id=qid,
        topic=qid.rsplit("-", 1)[0],
        question=f"Question {qid}?",
        answer="An answer.",
        relevant=[label(excerpt)],
        status=status,
        drafted_by="model@abc",
        prompt=PROMPT_VERSION,
        drafted_on=date(2026, 9, 24),
        source_chunk="a-0001",
    )


def test_a_run_records_the_rankings_the_settings_and_how_much_was_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # MLflow writes local artifacts under the working dir
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    chunks = pl.DataFrame(
        [
            {"chunk_id": "a-0001", "document_id": "a", "chunk": 1, "part": 1, "part_title": None,
             "text": "First light roast.", "characters": 18, "topics": ["roasting"],
             "topics_basis": "terms"},
            {"chunk_id": "a-0002", "document_id": "a", "chunk": 2, "part": 1, "part_title": None,
             "text": "Then a dark roast.", "characters": 18, "topics": ["roasting"],
             "topics_basis": "terms"},
        ],
        schema=CHUNKS_COLUMNS,
    )  # fmt: skip
    questions = [
        question("roasting-01", "Then a dark roast."),  # found second
        question("roasting-02", "A sentence the corpus lost."),  # its label is gone
        question("roasting-03", "First light roast.", status="rejected"),
    ]
    question_file = tmp_path / "questions.jsonl"
    question_file.write_text("the set", encoding="utf-8")
    config = domains.coffee.adapter().config

    run = evaluate_retrieval(
        config,
        "fixed",
        lambda query, k: [0, 1][:k],  # the same ranking for every question
        {"setting": 1.0},
        questions,
        question_file,
        chunks,
        ChunkingConfig(max_chars=1200, overlap_chars=200),
        tmp_path / "data",
        tracking_uri,
    )

    table = read_table(run.table.parent.parent)
    assert table["question_id"].to_list() == ["roasting-01", "roasting-02"]  # rejected: out
    assert table["retrieved"].to_list()[0] == ["a-0001", "a-0002"]
    assert table["labels_found"].to_list() == [1, 0]
    assert run.overall["reciprocal_rank"] == 0.25
    assert run.by_topic["recall_at_3_roasting"] == 0.5
    logged = MlflowClient(tracking_uri).get_run(run.run_id)
    assert logged.data.params["questions"] == "2"
    assert logged.data.params["questions_reviewed"] == "0"
    assert logged.data.params["labels_missing"] == "1"
    assert logged.data.params["setting"] == "1.0"
    assert logged.data.metrics["recall_at_3"] == 0.5
    experiment = mlflow.get_experiment(logged.info.experiment_id)
    assert experiment.name == experiment_name(config) == "coffee-retrieval"


def test_a_search_passes_only_if_it_beats_every_search_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    texts = [f"Passage number {n} about roast {n}." for n in range(30)]
    chunks = pl.DataFrame(
        [
            {"chunk_id": f"a-{n:04d}", "document_id": "a", "chunk": n + 1, "part": 1,
             "part_title": None, "text": text, "characters": len(text), "topics": ["roasting"],
             "topics_basis": "terms"}
            for n, text in enumerate(texts)
        ],
        schema=CHUNKS_COLUMNS,
    )  # fmt: skip
    questions = [question(f"roasting-{n:02d}", texts[n]) for n in range(30)]
    question_file = tmp_path / "questions.jsonl"
    question_file.write_text("the set", encoding="utf-8")
    config = domains.coffee.adapter().config
    chunking = ChunkingConfig(max_chars=1200, overlap_chars=200)

    def run(name: str, search: Callable[[str, int], list[int]], refs: dict[str, pl.DataFrame]):
        return evaluate_retrieval(
            config, name, search, {}, questions, question_file, chunks, chunking,
            tmp_path / "data", tracking_uri, references=refs,
        )  # fmt: skip

    def exact(query: str, k: int) -> list[int]:
        return [int(query.split("-")[1].rstrip("?"))][:k]

    worse = run("worse", lambda query, k: [], {})  # finds nothing
    better = run("better", exact, {"worse": worse.frame})

    assert better.comparisons["worse"].probability_better == 1.0
    assert better.passes
    assert not run("same", exact, {"worse": worse.frame, "better": better.frame}).passes
    logged = MlflowClient(tracking_uri).get_run(better.run_id)
    assert logged.data.tags["passes_gate"] == "True"
    assert logged.data.metrics["vs_worse_ndcg_at_10_probability_better"] == 1.0

    with pytest.raises(ValueError, match="same questions"):
        judged_against(better.frame, worse.frame.head(3))
