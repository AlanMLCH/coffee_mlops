"""The generator's benchmark: can the local model write the SQL and pick the tool?

Not a contest between models but a floor (CLAUDE.md, 2026-09-22): the agent's generator
must answer SQL questions right and route questions to the right tool. The bar was set
before any model was measured: 70% of the SQL questions answered right, with the two
repairs the agent allows, and 90% of the questions routed right.

SQL is judged by execution, as Spider and BIRD judge it: the answer, not the query's
text. Columns the question did not ask for are tolerated, row order is not judged, and
numbers are compared to four significant figures - so rounding an average is fine,
reporting a share as a percentage is not. A value per label may come laid out across,
one row with a column per label (`avg_washed`, `avg_natural`), instead of down: the
end-to-end evaluation found a right answer judged wrong for that alone.

The questions were written by the assistant that built this project, not by a person;
every reference query was run and its answer checked against figures established
earlier. Every run records which set it was scored on.
"""

import hashlib
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Self

import duckdb
import mlflow
import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator

from mlops_core.agent.prompts import ROUTER_VERSION, ROUTES, SQL_VERSION, Route, Tool
from mlops_core.agent.routing import route
from mlops_core.agent.sql import QueryResult, run_select
from mlops_core.agent.text_to_sql import Generator, write_sql
from mlops_core.config import DomainConfig
from mlops_core.provenance import code_version
from mlops_core.storage import write_table

SQL_BAR = 0.70
ROUTE_BAR = 0.90
# Deterministic, and a context that holds the data dictionary (about 3,300 tokens) with
# room for the question, a failed query and its error.
GENERATOR_OPTIONS = {"temperature": 0.0, "seed": 7, "num_ctx": 8192}
# The decided generator first; the alternative CLAUDE.md names if it misses the bar.
CANDIDATES = ("granite4.2:3b", "qwen3.5:4b")
COMPARED_ROWS = 1000  # an answer longer than this is not the reference's anyway
SQL_CASES_FILE = Path("evals") / "sql_questions.jsonl"
ROUTE_CASES_FILE = Path("evals") / "routing_questions.jsonl"
EVALUATIONS = "evaluations"


class SqlCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    sql: str  # the reference: its answer is the right one


class RouteCase(BaseModel):
    """A question and the tool it belongs to; the benchmark reads only that. The rest is
    what the end-to-end evaluation checks the agent's answer against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    route: Route
    tools: tuple[Tool, ...] = ()  # what a mixed question needs; one route needs its own tool
    sql: str | None = None  # a reference for the data part the SQL set does not hold
    model: str | None = None  # the model a prediction is for
    item: dict[str, str | float] = {}  # every field the question states, as the model spells it

    @model_validator(mode="after")
    def _expectations_fit_the_route(self) -> Self:
        if (self.route == "mixed") != bool(self.tools):
            raise ValueError(f"{self.id}: list the tools of a mixed question, and only then")
        if ("prediction" in self.needs) != bool(self.model and self.item):
            raise ValueError(f"{self.id}: a prediction states its model and item, and only it")
        return self

    @property
    def needs(self) -> tuple[Tool, ...]:
        """The tools a right answer needs."""
        return self.tools or (self.route,)  # type: ignore[return-value]


def load_cases[Case: BaseModel](path: Path, case: type[Case]) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [case.model_validate_json(line) for line in lines if line.strip()]


def same_answer(expected: QueryResult, got: QueryResult) -> bool:
    """Whether `got` holds the reference's answer: every reference column matched by a
    column of `got` with the same values, and the rows the same, in any order - or, for
    a value per label, the same values laid out across."""
    return _same_down(expected, got) or _same_across(expected, got)


def _same_down(expected: QueryResult, got: QueryResult) -> bool:
    if len(expected.rows) != len(got.rows):
        return False
    theirs = [_column(got, j) for j in range(len(got.columns))]
    mapping: list[int] = []
    for i in range(len(expected.columns)):
        wanted = sorted(_column(expected, i))
        match = next(
            (j for j, col in enumerate(theirs) if j not in mapping and sorted(col) == wanted),
            None,
        )
        if match is None:
            return False
        mapping.append(match)
    ours = sorted(tuple(_value(row[i]) for i in range(len(mapping))) for row in expected.rows)
    matched = sorted(tuple(_value(row[j]) for j in mapping) for row in got.rows)
    return ours == matched


def run_benchmark(
    generator: Generator,
    con: duckdb.DuckDBPyConnection,
    schema: str,
    context: dict[str, str],
    sql_cases: Sequence[SqlCase],
    route_cases: Sequence[RouteCase],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One row per SQL question and one per routing question, with the verdicts."""
    sql_rows = []
    for case in sql_cases:
        expected = run_select(con, case.sql, COMPARED_ROWS)
        start = time.perf_counter()
        answer = write_sql(generator, con, schema, case.question, COMPARED_ROWS)
        right = answer.result is not None and same_answer(expected, answer.result)
        sql_rows.append(
            {
                "case_id": case.id,
                "sql": answer.sql,
                "attempts": answer.attempts,
                "error": answer.error,
                "correct": right,
                "first_try": right and answer.attempts == 1,
                "seconds": time.perf_counter() - start,
            }
        )
    route_rows = []
    for route_case in route_cases:
        start = time.perf_counter()
        chosen = route(generator, context, route_case.question)
        route_rows.append(
            {
                "case_id": route_case.id,
                "expected": route_case.route,
                "chosen": chosen,
                "correct": chosen == route_case.route,
                "seconds": time.perf_counter() - start,
            }
        )
    return pl.DataFrame(sql_rows), pl.DataFrame(route_rows)


def summarise(sql: pl.DataFrame, routes: pl.DataFrame) -> dict[str, float]:
    """The accuracies, per route too, and the median seconds per question - the median,
    because the first question also pays for loading the model."""
    summary = {
        "sql_accuracy": _share(sql["correct"]),
        "sql_first_try": _share(sql["first_try"]),
        "sql_seconds": median(sql["seconds"].to_list()),
        "route_accuracy": _share(routes["correct"]),
        "route_seconds": median(routes["seconds"].to_list()),
    }
    for name in ROUTES:
        mine = routes.filter(pl.col("expected") == name)
        if mine.height:
            summary[f"route_accuracy_{name}"] = _share(mine["correct"])
    return summary


def meets_bar(summary: dict[str, float]) -> bool:
    return summary["sql_accuracy"] >= SQL_BAR and summary["route_accuracy"] >= ROUTE_BAR


def log_benchmark(
    config: DomainConfig,
    generator: str,
    sql: pl.DataFrame,
    routes: pl.DataFrame,
    case_files: Sequence[Path],
    data_dir: Path,
    tracking_uri: str,
    at: datetime | None = None,
) -> tuple[dict[str, float], str]:
    """Write the per-question verdicts to the evaluations layer and log one MLflow run:
    the generator, its options, the prompts' versions and which question sets."""
    summary = summarise(sql, routes)
    slug = "".join(c if c.isalnum() else "_" for c in generator.split("@")[0])
    tables = [
        write_table(frame, data_dir / EVALUATIONS / f"agent_{kind}_{slug}", {}, at)
        for kind, frame in (("sql", sql), ("routes", routes))
    ]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{config.name}-agent-benchmark")
    with mlflow.start_run(run_name=generator.split("@")[0]) as run:
        version = code_version()
        passed = meets_bar(summary)
        mlflow.set_tags({"meets_bar": str(passed), **(version.as_tags() if version else {})})
        params: dict[str, Any] = {
            "generator": generator,
            **{f"option_{k}": v for k, v in GENERATOR_OPTIONS.items()},
            "sql_prompt": SQL_VERSION,
            "router_prompt": ROUTER_VERSION,
            "sql_cases": sql.height,
            "route_cases": routes.height,
            "sql_bar": SQL_BAR,
            "route_bar": ROUTE_BAR,
            "cases_written_by": "assistant",
            "case_files_sha256": case_digest(case_files),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(summary)
        for table in tables:
            mlflow.log_artifact(str(table))
    return summary, run.info.run_id


def _same_across(expected: QueryResult, got: QueryResult) -> bool:
    """A (label, value) reference of several rows, answered as one row with a column per
    label: each label named by a column of its own that holds the label's value."""
    if len(expected.columns) != 2 or len(expected.rows) < 2 or len(got.rows) != 1:
        return False
    (row,) = got.rows
    names = [name.lower() for name in got.columns]
    used: set[int] = set()
    for label, value in expected.rows:
        match = next(
            (
                j
                for j, name in enumerate(names)
                if j not in used and str(label).lower() in name and _value(row[j]) == _value(value)
            ),
            None,
        )
        if match is None:
            return False
        used.add(match)
    return True


def _share(verdicts: pl.Series) -> float:
    right = verdicts.to_list()
    return sum(right) / len(right) if right else 0.0


def _column(result: QueryResult, index: int) -> list[str]:
    return [_value(row[index]) for row in result.rows]


def _value(value: object) -> str:
    """A cell as the comparison sees it: numbers to four significant figures."""
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, int | float):
        return f"{float(value):.4g}"
    return str(value).strip()


def case_digest(paths: Sequence[Path]) -> str:
    content = b"".join(path.read_bytes() for path in paths)
    return hashlib.sha256(content).hexdigest()[:12]
