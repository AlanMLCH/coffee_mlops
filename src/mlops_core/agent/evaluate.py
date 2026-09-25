"""The agent end to end: does a question get the right answer, from the right evidence?

The benchmark measured the generator's two small skills one at a time. This runs the
whole workflow - route, plan, tools, answer, verification - on the routing set, and
checks each answer against what is known to be right for its question, where it is:

- tools: the tools the question needs ran. The route is reported too, but a data
  question routed "mixed" whose plan still queried the tables got what it needed.
- verified: verification found nothing left - every figure from the evidence, every
  citation to evidence the tools returned.
- sql: the agent's query result holds the reference answer, judged as the benchmark
  judges it. The reference is the SQL set's query for the same question, or the case's
  own for the data part of a mixed one.
- passage: a passage the search returned contains an excerpt the retrieval set labels
  relevant to the same question.
- item: the prediction went to the right model, with every field the question stated,
  spelled as the model's vocabulary spells it rather than as the question did (the model
  would take the question's word for a category it never saw), and the API answered.

An answer is correct when every check that applies passes. What they cannot see - a
sentence a passage does not support - is left to a judge, admitted only after its
agreement with people is measured.

Each run is compared, question by question, with the previous one (paired bootstrap on
the same questions): a change to a prompt or a tool is judged as a model is.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import duckdb
import mlflow
import numpy as np
import polars as pl

from mlops_core.agent.benchmark import COMPARED_ROWS, EVALUATIONS, RouteCase, SqlCase, same_answer
from mlops_core.agent.graph import Reply
from mlops_core.agent.prompts import ROUTES
from mlops_core.agent.sql import run_select
from mlops_core.agent.tools import PredictionAnswer
from mlops_core.rag.questions import Question, contains
from mlops_core.stats import Comparison, compare
from mlops_core.storage import latest_partition, write_table

TABLE = "agent_answers"
CHECKS = ("sql_ok", "passage_ok", "item_ok")  # each applies to some questions only
SCHEMA: dict[str, Any] = {
    "case_id": pl.String,
    "route_expected": pl.String,
    "route": pl.String,
    "route_ok": pl.Boolean,
    "tools": pl.String,  # the tools that ran, comma-separated
    "tools_ok": pl.Boolean,
    "verified": pl.Boolean,
    "problems": pl.String,
    "sql_ok": pl.Boolean,
    "passage_ok": pl.Boolean,
    "item_ok": pl.Boolean,
    "item_errors": pl.String,  # the stated fields the request got wrong
    "correct": pl.Boolean,
    "text": pl.String,
    "sql": pl.String,
    "request": pl.String,
    "seconds": pl.Float64,
    "trace_id": pl.String,
}


@dataclass(frozen=True)
class Truth:
    """One question and what is known to be right for it."""

    case: RouteCase
    sql: str | None  # the reference query for its data part
    excerpts: tuple[str, ...]  # text a passage relevant to its knowledge part contains


def known_answers(
    cases: Sequence[RouteCase], sql_cases: Sequence[SqlCase], questions: Sequence[Question]
) -> list[Truth]:
    """Each case with the references the other sets hold for the same question: the
    routing set's data and knowledge questions were drawn from them, word for word."""
    references = {case.question: case.sql for case in sql_cases}
    labelled = {
        q.question: tuple(label.excerpt for label in q.relevant if label.grade > 0)
        for q in questions
    }
    return [
        Truth(
            case,
            (case.sql or references.get(case.question)) if "data" in case.needs else None,
            labelled.get(case.question, ()) if "knowledge" in case.needs else (),
        )
        for case in cases
    ]


def run_evaluation(
    ask: Callable[[str], Reply],
    truths: Sequence[Truth],
    con: duckdb.DuckDBPyConnection,
    trace_id: Callable[[], str | None] = mlflow.get_last_active_trace_id,
) -> pl.DataFrame:
    """Ask every question and check every answer: one row per question."""
    rows = []
    for truth in truths:
        start = time.perf_counter()
        reply = ask(truth.case.question)
        seconds = time.perf_counter() - start
        rows.append(check(truth, reply, con) | {"seconds": seconds, "trace_id": trace_id()})
    return pl.DataFrame(rows, schema=SCHEMA)


def check(truth: Truth, reply: Reply, con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Every check that applies to one answer, and whether it is correct."""
    case = truth.case
    ran = _tools_that_ran(reply)
    sql_ok = _sql_ok(truth.sql, reply, con) if truth.sql else None
    passage_ok = (
        any(contains(p["text"], e) for p in reply.passages for e in truth.excerpts)
        if truth.excerpts
        else None
    )
    item_errors = _item_errors(case, reply.prediction) if case.item else []
    item_ok = not item_errors if case.item else None
    tools_ok = set(case.needs) <= ran
    applicable = [ok for ok in (sql_ok, passage_ok, item_ok) if ok is not None]
    return {
        "case_id": case.id,
        "route_expected": case.route,
        "route": reply.route,
        "route_ok": reply.route == case.route,
        "tools": ",".join(sorted(ran)),
        "tools_ok": tools_ok,
        "verified": reply.verified,
        "problems": " | ".join(reply.problems),
        "sql_ok": sql_ok,
        "passage_ok": passage_ok,
        "item_ok": item_ok,
        "item_errors": ", ".join(item_errors),
        "correct": tools_ok and reply.verified and all(applicable),
        "text": reply.text,
        "sql": reply.sql.sql if reply.sql else None,
        "request": json.dumps(reply.prediction.request) if reply.prediction else None,
    }


def summarise(answers: pl.DataFrame) -> dict[str, float]:
    """The share of answers each check passed - of the questions it applies to - per
    route too, and the seconds per question: the median, and the slowest."""
    summary = {
        "correct": _share(answers["correct"]),
        "verified": _share(answers["verified"]),
        "route_accuracy": _share(answers["route_ok"]),
        "tools": _share(answers["tools_ok"]),
    }
    for name in CHECKS:
        applicable = answers[name].drop_nulls()
        if applicable.len():
            summary[name.removesuffix("_ok")] = _share(applicable)
    for name in ROUTES:
        mine = answers.filter(pl.col("route_expected") == name)
        if mine.height:
            summary[f"correct_{name}"] = _share(mine["correct"])
    summary["seconds_median"] = median(answers["seconds"].to_list())
    summary["seconds_max"] = float(answers["seconds"].max())  # type: ignore[arg-type]
    return summary


def versus(previous: pl.DataFrame, current: pl.DataFrame) -> Comparison | None:
    """How much more often the current run is correct than the previous one, on the
    questions both asked; None when they share none."""
    both = current.join(previous, on="case_id", suffix="_before")
    if both.is_empty():
        return None
    return compare(
        both["correct"].cast(pl.Float64).to_numpy(),
        both["correct_before"].cast(pl.Float64).to_numpy(),
        higher_is_better=True,
    )


def record(
    answers: pl.DataFrame, data_dir: Path, at: datetime | None = None
) -> tuple[Path, pl.DataFrame | None]:
    """Write this run's answers to the evaluations layer, and return the previous run's,
    read before the write, to compare with."""
    table_dir = data_dir / EVALUATIONS / TABLE
    before = latest_partition(table_dir)
    previous = pl.read_parquet(before / f"{TABLE}.parquet") if before else None
    return write_table(answers, table_dir, {}, at), previous


def log_evaluation(
    answers: pl.DataFrame,
    comparison: Comparison | None,
    params: Mapping[str, Any],
    table: Path,
) -> dict[str, float]:
    """Log the run's parameters, metrics and answers to the active MLflow run."""
    summary = summarise(answers)
    mlflow.log_params(dict(params))
    mlflow.log_metrics(summary | (comparison.as_metrics("vs_previous") if comparison else {}))
    mlflow.log_artifact(str(table))
    return summary


def _tools_that_ran(reply: Reply) -> set[str]:
    return (
        ({"data"} if reply.sql is not None else set())
        | ({"prediction"} if reply.prediction is not None else set())
        | ({"knowledge"} if reply.passages else set())
    )


def _sql_ok(reference: str, reply: Reply, con: duckdb.DuckDBPyConnection) -> bool:
    if reply.sql is None or reply.sql.result is None:
        return False
    return same_answer(run_select(con, reference, COMPARED_ROWS), reply.sql.result)


def _item_errors(case: RouteCase, prediction: PredictionAnswer | None) -> list[str]:
    """What the prediction got wrong: another model, a failed call, or a stated field
    missing or spelled otherwise."""
    if prediction is None:
        return ["no prediction"]
    errors = [] if prediction.model == case.model else [f"model {prediction.model}"]
    if prediction.response is None:
        errors.append("the API did not answer")
    for field, expected in case.item.items():
        got = prediction.request.get(field)
        if not _same(expected, got):
            errors.append(f"{field}={got!r}")
    return errors


def _same(expected: str | float, got: object) -> bool:
    if isinstance(expected, str):
        return isinstance(got, str) and got.strip().casefold() == expected.casefold()
    return isinstance(got, int | float) and not isinstance(got, bool) and got == expected


def _share(verdicts: pl.Series) -> float:
    return float(np.mean(verdicts.to_list())) if verdicts.len() else 0.0
