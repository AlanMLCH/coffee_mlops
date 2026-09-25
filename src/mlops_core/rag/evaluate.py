"""A retrieval evaluation: every question of the set searched, every ranking scored, one
table per search and one MLflow run per evaluation.

The table keeps each question's ranking and grades, not only the averages, because the
next search will be compared with this one question by question - a paired comparison
on the same questions is what a gate can trust - and because a per-topic average hides
which questions fail.

Rejected questions are left out; drafts are not. Whether a person reviewed them is
recorded with every run (`questions_reviewed`), so a number is never read as more
checked than it was.

A search earns its place the way a model does: against every search before it on the
ladder, paired on the same questions and bootstrapped, it has to be better on nDCG@10
with 95% certainty. nDCG@10 because it counts both whether the answer is found and how
high, in the ten chunks an answer can be built from.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from mlops_core.config import CHUNKS_TABLE, ChunkingConfig, DomainConfig
from mlops_core.provenance import code_version
from mlops_core.rag.metrics import CUTOFFS, grades, ranking_metrics
from mlops_core.rag.questions import Question, contains
from mlops_core.stats import Comparison, compare
from mlops_core.storage import latest_partition, write_table

# A search takes a question and how many chunks to return, and gives their positions in
# the chunk table, best first.
Search = Callable[[str, int], list[int]]

EVALUATIONS = "evaluations"  # the layer, beside clean, features and predictions
METRIC_PREFIXES = ("recall", "ndcg", "reciprocal", "average")
GATE_METRIC = "ndcg_at_10"
MIN_PROBABILITY_BETTER = 0.95
RESAMPLES = 5000


@dataclass(frozen=True)
class RetrievalRun:
    table: Path  # one row per question
    frame: pl.DataFrame  # the same table, for the searches after this one to be judged by
    overall: dict[str, float]  # the mean of each metric
    by_topic: dict[str, float]  # the same, per topic: `<metric>_<topic>`
    comparisons: dict[str, Comparison]  # against each search before it, on GATE_METRIC
    run_id: str

    @property
    def passes(self) -> bool:
        """Better than every search before it, with the certainty the gate asks for."""
        return all(
            c.probability_better >= MIN_PROBABILITY_BETTER for c in self.comparisons.values()
        )


def experiment_name(config: DomainConfig) -> str:
    """One experiment for every search of a domain: they answer the same questions."""
    return f"{config.name}-retrieval"


def score_search(
    search: Search, questions: Sequence[Question], chunks: pl.DataFrame
) -> pl.DataFrame:
    """Each question searched and its ranking graded against its labels."""
    ids = chunks["chunk_id"].to_list()
    places = list(zip(chunks["document_id"].to_list(), chunks["text"].to_list(), strict=True))
    rows = []
    for question in questions:
        found = search(question.question, max(CUTOFFS))
        earned = grades([places[p] for p in found], question.relevant)
        rows.append(
            {
                "question_id": question.id,
                "topic": question.topic,
                "status": question.status,
                "retrieved": [ids[p] for p in found],
                "grades": earned,
                # Labels the current corpus still holds: a cut that split an excerpt would
                # otherwise pass for a search that missed it.
                "labels_found": sum(
                    any(d == label.document_id and contains(t, label.excerpt) for d, t in places)
                    for label in question.relevant
                ),
                **ranking_metrics(earned, question.relevant),
            }
        )
    return pl.DataFrame(rows)


def summarise(table: pl.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """The mean of every metric, overall and per topic (`<metric>_<topic>`)."""
    metrics = [c for c in table.columns if c.startswith(METRIC_PREFIXES)]
    overall = table.select(pl.col(metrics).mean()).row(0, named=True)
    by_topic = table.group_by("topic").agg(pl.col(metrics).mean())
    per_topic = {
        f"{metric}_{row['topic']}": row[metric]
        for row in by_topic.iter_rows(named=True)
        for metric in metrics
    }
    return (
        {name: float(value) for name, value in overall.items()},
        {name: float(value) for name, value in per_topic.items()},
    )


def judged_against(frame: pl.DataFrame, reference: pl.DataFrame) -> Comparison:
    """A paired bootstrap of GATE_METRIC, question by question."""
    paired = frame.join(reference, on="question_id", suffix="_reference")
    if paired.height != frame.height:
        raise ValueError("Searches can only be compared on the same questions")
    return compare(
        paired[GATE_METRIC].to_numpy().astype(np.float64),
        paired[f"{GATE_METRIC}_reference"].to_numpy().astype(np.float64),
        resamples=RESAMPLES,
        higher_is_better=True,
    )


def evaluate_retrieval(
    config: DomainConfig,
    name: str,
    search: Search,
    settings: Mapping[str, str | float],
    questions: Sequence[Question],
    question_file: Path,
    chunks: pl.DataFrame,
    chunking: ChunkingConfig,
    data_dir: Path,
    tracking_uri: str,
    references: Mapping[str, pl.DataFrame] | None = None,
    at: datetime | None = None,
) -> RetrievalRun:
    """Score one search on every question nobody rejected, judge it against the
    `references` (the searches before it), write the per-question table to
    `evaluations/retrieval_<name>` and log the run: the search's settings, the cut,
    which question set it was, how much of it a person checked, and the verdicts."""
    evaluated = [q for q in questions if q.status != "rejected"]
    table = score_search(search, evaluated, chunks)
    overall, by_topic = summarise(table)
    comparisons = {
        reference: judged_against(table, frame) for reference, frame in (references or {}).items()
    }
    chunks_partition = latest_partition(data_dir / "clean" / CHUNKS_TABLE)
    lineage = {CHUNKS_TABLE: chunks_partition.name} if chunks_partition else {}
    path = write_table(table, data_dir / EVALUATIONS / f"retrieval_{name}", lineage, at)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name(config))
    with mlflow.start_run(run_name=name) as run:
        version = code_version()
        mlflow.set_tags({"search": name, **(version.as_tags() if version else {})})
        mlflow.log_params(
            {
                "search": name,
                **settings,
                "chunk_max_chars": chunking.max_chars,
                "chunk_overlap_chars": chunking.overlap_chars,
                "chunks": chunks.height,
                "questions": len(evaluated),
                "questions_reviewed": sum(q.status in ("accepted", "edited") for q in evaluated),
                # Which set, exactly: a reviewed set is a different yardstick.
                "question_set_sha256": hashlib.sha256(question_file.read_bytes()).hexdigest()[:12],
                "labels_missing": int(
                    sum(len(q.relevant) for q in evaluated) - table["labels_found"].sum()
                ),
            }
        )
        mlflow.log_metrics(overall | by_topic)
        for reference, comparison in comparisons.items():
            mlflow.log_metrics(comparison.as_metrics(f"vs_{reference}_{GATE_METRIC}"))
        verdict = RetrievalRun(path, table, overall, by_topic, comparisons, run.info.run_id)
        mlflow.set_tag("passes_gate", str(verdict.passes))
        mlflow.log_artifact(str(path))
    return verdict
