"""Text to SQL, routing, and the benchmark that measures the generator driving them.

A scripted generator stands in for the model: what is under test is the loop around it
(what it is shown, how an error goes back to it, how its answer is judged), not a model.
"""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import httpx
import polars as pl
import pytest
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from typer.testing import CliRunner

import domains.coffee
from mlops_core import cli
from mlops_core.adapter import domain_dir
from mlops_core.agent.benchmark import (
    ROUTE_CASES_FILE,
    SQL_CASES_FILE,
    RouteCase,
    SqlCase,
    load_cases,
    log_benchmark,
    meets_bar,
    run_benchmark,
    same_answer,
)
from mlops_core.agent.dictionary import dictionary_path
from mlops_core.agent.prompts import ROUTES, RouteReply, SqlReply
from mlops_core.agent.routing import route, routing_context
from mlops_core.agent.sql import QueryResult, read_only
from mlops_core.agent.text_to_sql import MAX_ATTEMPTS, write_sql
from mlops_core.rag import llm
from mlops_core.rag.llm import ollama_client
from mlops_core.storage import read_table, write_table

OLLAMA = Path(__file__).parent / "fixtures" / "ollama"


class Scripted:
    """A generator that answers from a script and remembers every prompt."""

    def __init__(self, answer: Callable[[str], dict[str, Any]]):
        self.answer = answer
        self.prompts: list[str] = []

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        self.prompts.append(prompt)
        return reply.model_validate(self.answer(prompt))


@pytest.fixture
def session(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    data = tmp_path / "coffee"
    write_table(pl.DataFrame({"country": ["A", "B", "B"], "points": [80.0, 85.5, 90.25]}),
                data / "clean" / "lots", {})  # fmt: skip
    return read_only(data)


def result(columns: list[str], rows: list[tuple[Any, ...]]) -> QueryResult:
    return QueryResult("SELECT", columns, rows, False)


# --- Judging an answer ---------------------------------------------------------------------


def test_an_answer_is_judged_by_its_values_not_its_shape() -> None:
    expected = result(["country", "points"], [("B", 87.875), ("A", 80.0)])

    assert same_answer(expected, result(["country", "avg"], [("A", 80.0), ("B", 87.875)]))
    # Rounded, with an extra column and the columns swapped: still the answer.
    assert same_answer(expected, result(["n", "p", "c"], [(2, 87.88, "B"), (1, 80.0, "A")]))
    assert not same_answer(expected, result(["country"], [("B",), ("A",)]))  # a column short
    assert not same_answer(expected, result(["c", "p"], [("B", 87.9), ("A", 80.0), ("C", 1.0)]))
    assert not same_answer(expected, result(["c", "p"], [("B", 80.0), ("A", 87.875)]))


def test_a_value_per_label_may_come_laid_out_across() -> None:
    """One row, a column per label: the same two averages as two rows."""
    expected = result(["method", "points"], [("washed", 82.187), ("natural", 82.563)])

    assert same_answer(expected, result(["avg_washed", "avg_natural"], [(82.19, 82.563)]))
    assert same_answer(expected, result(["natural_avg", "washed_avg"], [(82.563, 82.187)]))
    assert not same_answer(expected, result(["avg_washed", "avg_natural"], [(82.19, 82.19)]))
    assert not same_answer(expected, result(["a", "b"], [(82.187, 82.563)]))  # labels unnamed
    one = result(["method", "points"], [("washed", 82.187)])
    assert not same_answer(one, result(["avg_washed"], [(82.187,)]))  # one row is not a layout


def test_a_share_is_not_a_percentage() -> None:
    assert not same_answer(result(["s"], [(0.6772,)]), result(["pct"], [(67.72,)]))


def test_nulls_and_flags_are_values_like_any_other() -> None:
    assert same_answer(result(["x"], [(None,), (True,)]), result(["y"], [(True,), (None,)]))
    assert not same_answer(result(["x"], [(None,)]), result(["y"], [(0,)]))


# --- Writing SQL ---------------------------------------------------------------------------


def test_a_failed_query_goes_back_with_its_error_until_one_runs(
    session: duckdb.DuckDBPyConnection,
) -> None:
    queries = iter(["SELECT nope FROM clean.lots", "SELECT count(*) AS n FROM clean.lots"])
    model = Scripted(lambda prompt: {"sql": next(queries)})

    answer = write_sql(model, session, "## `clean.lots`", "How many lots?")

    assert answer.attempts == 2
    assert answer.result is not None and answer.result.rows == [(3,)]
    assert "## `clean.lots`" in model.prompts[0]
    assert "SELECT nope FROM clean.lots" in model.prompts[1]
    assert 'Referenced column "nope" not found' in model.prompts[1]


def test_a_model_that_never_writes_a_query_that_runs_gives_up_with_the_error(
    session: duckdb.DuckDBPyConnection,
) -> None:
    model = Scripted(lambda prompt: {"sql": "DROP VIEW clean.lots"})

    answer = write_sql(model, session, "", "Drop it")

    assert (answer.result, answer.attempts) == (None, MAX_ATTEMPTS)
    assert answer.error == "Only SELECT may run; this is DROP"


# --- Routing -------------------------------------------------------------------------------


def test_the_router_is_told_what_each_tool_covers_in_the_domains_terms() -> None:
    config = domains.coffee.adapter().config
    dictionary = dictionary_path(domain_dir("coffee")).read_text(encoding="utf-8")

    built = {"clean.coffee_reviews", "features.review_features"}
    context = routing_context(config, dictionary, built)
    model = Scripted(lambda prompt: {"route": "prediction"})

    assert route(model, context, "What would it score?") == "prediction"
    prompt = model.prompts[0]
    assert "clean.coffee_reviews — one graded lot" in prompt
    assert "clean.market_context" not in prompt  # not built, not offered
    assert "review_features" not in prompt  # a model's inputs, never offered
    assert f"review: {config.model_named('review').description}" in prompt
    assert "roasting: The roast" in prompt
    assert "Question: What would it score?" in prompt


# --- The benchmark -------------------------------------------------------------------------


def test_the_benchmark_scores_each_question_and_logs_the_verdicts(
    session: duckdb.DuckDBPyConnection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    sql_cases = [
        SqlCase(id="n", question="How many lots?", sql="SELECT count(*) FROM clean.lots"),
        SqlCase(id="top", question="Best?", sql="SELECT max(points) FROM clean.lots"),
    ]
    expects: dict[str, dict[str, Any]] = {  # what the end-to-end evaluation would check
        "prediction": {"model": "review", "item": {"country": "Kenya"}},
        "mixed": {"tools": ["data", "knowledge"]},
    }
    route_cases = [
        RouteCase.model_validate(
            {"id": f"r{n}", "question": f"{r}?", "route": r} | expects.get(r, {})
        )
        for n, r in enumerate(ROUTES)
    ]

    def answer(prompt: str) -> dict[str, Any]:
        if "Question: How many lots?" in prompt:
            return {"sql": "SELECT count(*) AS lots FROM clean.lots"}
        if "Question: Best?" in prompt:
            return {"sql": "SELECT min(points) FROM clean.lots"}  # runs, and is wrong
        return {"route": "data"}

    sql, routes = run_benchmark(Scripted(answer), session, "", {"subject": "x", "tables": "",
                                "models": "", "topics": ""}, sql_cases, route_cases)  # fmt: skip
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text("cases", encoding="utf-8")
    config = domains.coffee.adapter().config
    summary, run_id = log_benchmark(
        config, "model:1b@abc", sql, routes, [case_file], tmp_path / "data", tracking_uri
    )

    assert sql["correct"].to_list() == [True, False]
    assert summary["sql_accuracy"] == 0.5
    assert summary["route_accuracy"] == 0.25
    assert summary["route_accuracy_data"] == 1.0
    assert not meets_bar(summary)
    logged = MlflowClient(tracking_uri).get_run(run_id)
    assert logged.data.tags["meets_bar"] == "False"
    assert logged.data.params["generator"] == "model:1b@abc"
    assert logged.data.params["cases_written_by"] == "assistant"
    saved = read_table(tmp_path / "data" / "evaluations" / "agent_routes_model_1b")
    assert saved["chosen"].to_list() == ["data"] * 4


def test_the_committed_cases_are_one_select_each_and_route_somewhere_known() -> None:
    home = domain_dir("coffee")
    sql_cases = load_cases(home / SQL_CASES_FILE, SqlCase)
    route_cases = load_cases(home / ROUTE_CASES_FILE, RouteCase)
    parse = duckdb.connect()

    assert len(sql_cases) >= 20
    for case in sql_cases:
        (statement,) = parse.extract_statements(case.sql)
        assert statement.type == duckdb.StatementType.SELECT, case.id
    assert {case.route for case in route_cases} == set(ROUTES)
    assert len({case.id for case in route_cases}) == len(route_cases)


def test_the_command_runs_each_generator_and_reports_the_bar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With an Ollama that answers every SQL question "SELECT 1" and every question
    "data": the plumbing, not a model, is under test."""
    clean = tmp_path / "data" / "coffee" / "clean"
    write_table(pl.DataFrame({"x": [1]}), clean / "coffee_reviews", {})
    home = tmp_path / "domain"
    (home / "evals").mkdir(parents=True)
    (home / "data_dictionary.md").write_text("## `clean.coffee_reviews` — lots\n", "utf-8")
    (home / SQL_CASES_FILE).write_text(
        SqlCase(id="one", question="One?", sql="SELECT 1").model_dump_json() + "\n", "utf-8"
    )
    (home / ROUTE_CASES_FILE).write_text(
        RouteCase(id="d", question="Data?", route="data").model_dump_json() + "\n", "utf-8"
    )
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "domain_dir", lambda domain: home)
    monkeypatch.chdir(tmp_path)
    tags = json.loads((OLLAMA / "tags.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        shape = json.loads(request.content)["format"]["properties"]
        reply = SqlReply(sql="SELECT 1") if "sql" in shape else RouteReply(route="data")
        return httpx.Response(200, json={"message": {"content": reply.model_dump_json()}})

    @contextmanager
    def served(url: str) -> Iterator[httpx.Client]:
        with ollama_client(url, httpx.MockTransport(handler)) as c:
            yield c

    monkeypatch.setattr(llm, "ollama_client", served)

    result = CliRunner().invoke(cli.app, ["agent", "benchmark", "--generator", "qwen3.5:4b"])

    assert result.exit_code == 0, result.output
    assert "qwen3.5:4b@2a654d98e6fb: SQL 100% right" in result.output
    assert "meets the bar" in result.output
