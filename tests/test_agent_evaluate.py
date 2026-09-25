"""The end-to-end evaluation: which checks apply to a question, what makes an answer
correct, and how two runs are compared.

Replies are built by hand: what is under test is the judging, not the agent.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import mlflow
import polars as pl
import pytest
from pydantic import ValidationError

from mlops_core.agent.benchmark import RouteCase, SqlCase
from mlops_core.agent.evaluate import (
    SCHEMA,
    TABLE,
    Truth,
    check,
    known_answers,
    log_evaluation,
    record,
    run_evaluation,
    summarise,
    versus,
)
from mlops_core.agent.graph import Reply
from mlops_core.agent.sql import QueryResult, read_only
from mlops_core.agent.text_to_sql import SqlAnswer
from mlops_core.agent.tools import PredictionAnswer
from mlops_core.rag.questions import Judgment, Question
from mlops_core.storage import write_table

TOP_STATE = "SELECT state FROM clean.mexico_production ORDER BY production_t DESC LIMIT 1"
EXCERPT = "Cooler temperatures at altitude delay ripening"
PASSAGE = {"chunk_id": "fao-0001", "text": f"{EXCERPT}, and acidity develops."}
PREDICTED = {"target": "total_cup_points", "prediction": 84.4, "model_version": "1"}


@pytest.fixture
def session(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    data = tmp_path / "coffee"
    write_table(pl.DataFrame({"state": ["Chiapas", "Puebla"], "production_t": [391690.56, 1.0]}),
                data / "clean" / "mexico_production", {})  # fmt: skip
    return read_only(data)


def case(id: str, route: str, **expects: Any) -> RouteCase:
    return RouteCase.model_validate({"id": id, "question": f"{id}?", "route": route, **expects})


def reply(
    route: str = "data",
    rows: list[tuple[Any, ...]] | None = None,
    prediction: PredictionAnswer | None = None,
    passages: list[dict[str, Any]] | None = None,
    problems: list[str] | None = None,
) -> Reply:
    sql = (
        SqlAnswer("SELECT ...", QueryResult("SELECT ...", ["state"], rows, False), None, 1)
        if rows is not None
        else None
    )
    return Reply("q", route, "An answer [sql].", [], sql, prediction, passages or [],  # type: ignore[arg-type]
                 problems or [])  # fmt: skip


def question(text: str, *excerpts: tuple[str, int]) -> Question:
    return Question(
        id="cultivation-01", topic="cultivation", question=text, answer="Because.",
        relevant=[Judgment(document_id="fao", part=1, excerpt=e, grade=g) for e, g in excerpts],  # type: ignore[arg-type]
        status="draft", drafted_by="m@1", prompt="p", drafted_on=datetime(2026, 9, 24).date(),
        source_chunk="fao-0001",
    )  # fmt: skip


# --- The cases ----------------------------------------------------------------------------


def test_a_case_says_what_it_needs_and_nothing_its_route_does_not() -> None:
    assert case("m", "mixed", tools=["data", "knowledge"]).needs == ("data", "knowledge")
    assert case("d", "data").needs == ("data",)
    with pytest.raises(ValidationError, match="list the tools of a mixed question"):
        case("m", "mixed")
    with pytest.raises(ValidationError, match="list the tools of a mixed question"):
        case("d", "data", tools=["data"])
    with pytest.raises(ValidationError, match="a prediction states its model and item"):
        case("p", "prediction", model="review")
    with pytest.raises(ValidationError, match="a prediction states its model and item"):
        case("d", "data", model="review", item={"country": "Kenya"})


def test_references_come_from_the_other_sets_by_the_questions_words() -> None:
    cases = [
        case("data-01", "data"),
        case("knowledge-01", "knowledge"),
        case("mixed-01", "mixed", tools=["data", "knowledge"], sql="SELECT 1"),
        case("prediction-01", "prediction", model="review", item={"country": "Kenya"}),
    ]
    sql_cases = [SqlCase(id="production-01", question="data-01?", sql=TOP_STATE)]
    labelled = [question("knowledge-01?", (EXCERPT, 2), ("Judged and not relevant", 0))]

    truths = known_answers(cases, sql_cases, labelled)

    assert [(t.sql, t.excerpts) for t in truths] == [
        (TOP_STATE, ()),
        (None, (EXCERPT,)),  # a label graded 0 is not an answer
        ("SELECT 1", ()),  # its own reference; no labelled passage for its knowledge part
        (None, ()),
    ]


# --- The checks ---------------------------------------------------------------------------


def test_a_data_answer_is_right_when_its_query_holds_the_reference_answer(
    session: duckdb.DuckDBPyConnection,
) -> None:
    truth = Truth(case("data-01", "data"), TOP_STATE, ())

    right = check(truth, reply(rows=[("Chiapas",)]), session)
    wrong = check(truth, reply(rows=[("Puebla",)]), session)
    unsure = reply(rows=[("Chiapas",)], problems=["The figure 9 is not..."])
    unverified = check(truth, unsure, session)
    no_query = check(truth, reply(route="knowledge", passages=[PASSAGE]), session)

    assert (right["sql_ok"], right["correct"], right["passage_ok"], right["item_ok"]) == (
        True, True, None, None,
    )  # fmt: skip
    assert (wrong["sql_ok"], wrong["correct"]) == (False, False)
    assert (unverified["verified"], unverified["correct"]) == (False, False)
    assert unverified["problems"] == "The figure 9 is not..."
    assert (no_query["route_ok"], no_query["tools_ok"], no_query["sql_ok"]) == (False, False, False)


def test_a_mixed_question_needs_every_tool_whatever_the_route(
    session: duckdb.DuckDBPyConnection,
) -> None:
    truth = Truth(case("mixed-01", "mixed", tools=["data", "knowledge"]), None, (EXCERPT,))

    both = check(truth, reply(route="data", rows=[("Chiapas",)], passages=[PASSAGE]), session)
    half = check(truth, reply(route="mixed", passages=[PASSAGE]), session)
    elsewhere = check(truth, reply(route="mixed", rows=[], passages=[{"text": "Other."}]), session)

    assert (both["route_ok"], both["tools_ok"], both["tools"], both["correct"]) == (
        False, True, "data,knowledge", True,
    )  # fmt: skip
    assert (half["tools_ok"], half["correct"]) == (False, False)
    assert (elsewhere["passage_ok"], elsewhere["correct"]) == (False, False)


def test_a_prediction_is_right_only_with_every_stated_field_as_the_model_spells_it(
    session: duckdb.DuckDBPyConnection,
) -> None:
    truth = Truth(
        case("p", "prediction", model="review",
             item={"country": "Indonesia", "processing_method": "semi_washed", "altitude_m": 1500}),
        None, (),
    )  # fmt: skip
    request = {"country": "indonesia", "processing_method": "semi_washed", "altitude_m": 1500.0}

    def predicted(model: str = "review", response: dict[str, Any] | None = PREDICTED,
                  **changed: Any) -> PredictionAnswer:  # fmt: skip
        return PredictionAnswer(model, request | changed, response, None)

    right = check(truth, reply("prediction", prediction=predicted()), session)
    hyphen = predicted(processing_method="semi-washed")
    spelled = check(truth, reply("prediction", prediction=hyphen), session)
    missing = check(truth, reply("prediction", prediction=predicted(altitude_m=None)), session)
    other = check(truth, reply("prediction", prediction=predicted("offer", None)), session)
    none = check(truth, reply("prediction"), session)

    assert (right["item_ok"], right["correct"], right["item_errors"]) == (True, True, "")
    assert '"country": "indonesia"' in right["request"]  # the request, as it was sent
    assert spelled["item_errors"] == "processing_method='semi-washed'"
    assert missing["item_errors"] == "altitude_m=None"
    assert other["item_errors"] == "model offer, the API did not answer"
    assert (none["item_errors"], none["tools_ok"]) == ("no prediction", False)
    assert not any(row["correct"] for row in (spelled, missing, other, none))


# --- Runs ---------------------------------------------------------------------------------


def answers(*correct: bool, route: str = "data") -> pl.DataFrame:
    rows = [
        {"case_id": f"c{n}", "route_expected": route, "route": route, "route_ok": True,
         "tools_ok": True, "verified": ok, "sql_ok": ok if n % 2 else None, "passage_ok": None,
         "item_ok": None, "correct": ok, "seconds": float(n + 1)}
        for n, ok in enumerate(correct)
    ]  # fmt: skip
    return pl.DataFrame(rows, schema={k: SCHEMA[k] for k in rows[0]})


def test_the_summary_counts_each_check_over_the_questions_it_applies_to() -> None:
    runs = [answers(True, False, True, True), answers(False, route="mixed")]
    summary = summarise(pl.concat(runs))

    assert summary["correct"] == pytest.approx(3 / 5)
    assert summary["sql"] == 0.5  # c1 and c3 carry a reference: one right
    assert "passage" not in summary and "item" not in summary
    assert (summary["correct_data"], summary["correct_mixed"]) == (0.75, 0.0)
    assert (summary["seconds_median"], summary["seconds_max"]) == (2.0, 4.0)


def test_a_run_is_compared_with_the_previous_on_the_questions_both_asked() -> None:
    before = answers(False, False, True, False)
    after = answers(True, True, True, False)

    better = versus(before, after)

    assert better is not None
    assert better.difference == 0.5
    assert better.probability_better > 0.9
    assert versus(before, after.with_columns(pl.lit("x").alias("case_id"))) is None


def test_record_hands_back_the_previous_run(tmp_path: Path) -> None:
    first, none = record(answers(True), tmp_path, datetime(2026, 9, 25, 10, tzinfo=UTC))
    second, previous = record(answers(False), tmp_path, datetime(2026, 9, 25, 11, tzinfo=UTC))

    assert none is None
    assert previous is not None and previous["correct"].to_list() == [True]
    assert first.parent.parent == second.parent.parent == tmp_path / "evaluations" / TABLE


def test_every_question_is_asked_timed_and_traced(session: duckdb.DuckDBPyConnection) -> None:
    truths = [
        Truth(case("data-01", "data"), TOP_STATE, ()),
        Truth(case("data-02", "data"), None, ()),
    ]
    asked: list[str] = []

    def ask(text: str) -> Reply:
        asked.append(text)
        return reply(rows=[("Chiapas",)])

    table = run_evaluation(ask, truths, session, trace_id=lambda: "tr-1")

    assert asked == ["data-01?", "data-02?"]
    assert table.schema == pl.Schema(SCHEMA)
    assert table["sql_ok"].to_list() == [True, None]
    assert table["trace_id"].to_list() == ["tr-1", "tr-1"]


def test_the_run_is_logged_with_its_comparison(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("coffee-agent-eval")
    table, _ = record(answers(True, False), tmp_path)
    comparison = versus(answers(False, False), answers(True, False))

    with mlflow.start_run() as run:
        summary = log_evaluation(answers(True, False), comparison, {"generator": "m@1"}, table)

    logged = mlflow.get_run(run.info.run_id).data
    assert logged.params == {"generator": "m@1"}
    assert logged.metrics["correct"] == summary["correct"] == 0.5
    assert logged.metrics["vs_previous_difference"] == 0.5
