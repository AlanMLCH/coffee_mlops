"""The agent: its steps, its checks, and the record it leaves.

A scripted generator stands in for the model, answering by the shape it is asked for;
the prediction API is a mock transport; retrieval is a function returning passages.
What is under test is the workflow - which step runs, what each is shown, when an
answer is written again and which one is kept - not a model.
"""

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import httpx
import mlflow
import numpy as np
import polars as pl
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel
from qdrant_client import QdrantClient
from typer.testing import CliRunner

import domains.coffee
from mlops_core import cli
from mlops_core.adapter import domain_dir, load_adapter
from mlops_core.agent import registry
from mlops_core.agent.graph import Agent, Reply
from mlops_core.agent.planner import StepResult
from mlops_core.agent.prompts import PROMPTS, PlannedStep
from mlops_core.agent.registry import PREFIX, register_prompts
from mlops_core.agent.sql import QueryResult, read_only, views
from mlops_core.agent.study_cards import StudyCard, StudyFinder
from mlops_core.agent.text_to_sql import (
    NO_TABLE,
    PROSE,
    SqlAnswer,
    absent_values,
    counted_nothing,
    repair_hint,
    same_values,
    unasked_values,
    voted,
    write_sql,
    writes_prose,
)
from mlops_core.agent.tools import PredictionAnswer, predict
from mlops_core.agent.verify import cited_ids, figures, problems
from mlops_core.config import CHUNKS_TABLE, DOCUMENTS_TABLE, Settings, SqlGuard
from mlops_core.data.corpus import CHUNKS_COLUMNS
from mlops_core.rag import llm
from mlops_core.rag.llm import ollama_client
from mlops_core.rag.vectors import build_index, chunks_digest, embedding_table
from mlops_core.storage import latest_partition, write_table

OLLAMA = Path(__file__).parent / "fixtures" / "ollama"
TOP = "SELECT state FROM clean.mexico_production ORDER BY production_t DESC LIMIT 1"
PASSAGE = {
    "chunk_id": "doc-0001",
    "document_id": "fao",
    "part": 12,
    "part_title": None,
    "text": "Cooler temperatures at altitude delay ripening, and acidity develops.",
}
PREDICTED = {
    "target": "price_mxn_per_kg",
    "prediction": 1324.93,
    "model_version": "5",
    "model_source": "registry",
    "context": {},
}


# What a scripted model says when a test does not script it: the second opinion adds no
# tool, and an answer answers.
NOTHING_TO_ADD = {"predicts": False, "figures": False}


# The prediction tools of the models added in v1.1.1, in the YAML's order.
NEW_PREDICTIONS = [
    "predict_zones",
    "predict_green_range",
    "predict_shelf_price",
    "predict_shop_kind",
    "predict_auction",
    "predict_households",
]


def answering(reply: dict[str, Any], shape: str) -> dict[str, Any]:
    return {"answered": True} | reply if shape == "AnswerReply" else reply


class Scripted:
    """Answers each call by the name of the shape it is asked for."""

    def __init__(self, **answers: Callable[[str], dict[str, Any]] | list[dict[str, Any]]):
        self.answers = {k: iter(v) if isinstance(v, list) else v for k, v in answers.items()}
        self.answers.setdefault("NeedsReply", lambda prompt: NOTHING_TO_ADD)
        self.prompts: list[tuple[str, str]] = []

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        self.prompts.append((reply.__name__, prompt))
        answer = self.answers[reply.__name__]
        given = next(answer) if isinstance(answer, Iterator) else answer(prompt)
        return reply.model_validate(answering(given, reply.__name__))

    def asked(self, shape: str) -> list[str]:
        return [prompt for name, prompt in self.prompts if name == shape]


def api(respond: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(respond))


@pytest.fixture
def session(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    data = tmp_path / "coffee"
    write_table(pl.DataFrame({"state": ["Chiapas", "Puebla"], "production_t": [391690.56, 1.0]}),
                data / "clean" / "mexico_production", {})  # fmt: skip
    return read_only(data)


def agent(
    generator: Scripted,
    session: duckdb.DuckDBPyConnection,
    responder: Callable[[httpx.Request], httpx.Response] | None = None,
    voters: Sequence[Scripted] = (),
    library: bool = True,
    studies: StudyFinder | None = None,
    check_result: bool = False,
    planner: bool = False,
) -> Agent:
    return Agent(
        generator,
        domains.coffee.adapter(),
        session,
        "## `clean.mexico_production` — coffee grown",
        {"subject": "coffee", "tables": "", "models": "  - review: a cup score", "topics": ""},
        lambda question, k: [PASSAGE],
        api(responder or (lambda request: httpx.Response(200, json=PREDICTED))),
        {"fao": {"title": "Arabica coffee manual", "publisher": "FAO", "year": 2005}},
        {"answer": "prompts:/mlops-agent-answer/1"},
        voters=list(voters),
        library=library,
        studies=studies,
        check_result=check_result,
        planner=planner,
    )


# --- Verification --------------------------------------------------------------------------


def test_figures_are_read_as_written_and_citations_are_not_figures() -> None:
    assert figures("Chiapas made 391,690.56 t [c2], 37% of 2025.") == [
        ("391,690.56", 391690.56, 2, False),
        ("37%", 37.0, 0, True),
        ("2025", 2025.0, 0, False),
    ]


def test_an_answer_may_round_and_restate_a_share_as_a_percentage() -> None:
    evidence = "share | 0.3662\nproduction | 391690.56"

    assert problems("About 36.6% [sql], or 391,691 t [sql].", ["sql"], evidence, ["sql"]) == []


def test_what_verification_catches() -> None:
    evidence = "production | 391690.56"

    found = problems("It made 500,000 t [c9].", ["[c9]"], evidence, ["sql", "c1"])

    assert found == [
        "Cited c9, which is not evidence you were given.",
        "The figure 500,000 is not in the evidence; use only figures it gives.",
    ]
    assert problems("Chiapas.", [], evidence, ["sql"]) == [
        "Cite the evidence each statement comes from, by its id in brackets."
    ]
    assert cited_ids("As [c1] says", [" [sql] ", ""]) == {"c1", "sql"}


# --- The prediction tool -------------------------------------------------------------------


def test_a_prediction_is_the_model_chosen_and_the_item_it_describes() -> None:
    generator = Scripted(
        ModelChoice=lambda prompt: {"model": "offer"},
        Offer=lambda prompt: {"shop": "Almanegra", "bag_grams": 250, "variety": "Gesha"},
    )
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=PREDICTED)

    answer = predict(generator, domains.coffee.adapter(), api(respond), "A Gesha at Almanegra?")

    assert answer.model == "offer" and answer.error is None
    # The closed vocabularies are lower case, whatever the model wrote.
    assert sent == [{"shop": "almanegra", "bag_grams": 250.0, "variety": "gesha"}]
    choice = generator.asked("ModelChoice")[0]
    assert "- offer (predicts price_mxn_per_kg): The price" in choice
    assert "Asked with: shop, bag_grams, country" in choice  # the request's inputs
    assert "A model predicts The price per kilogram" in generator.asked("Offer")[0]


def test_the_choice_is_among_the_models_the_router_was_shown() -> None:
    generator = Scripted(
        ModelChoice=lambda prompt: {"model": "offer"},
        Offer=lambda prompt: {"shop": "almanegra", "bag_grams": 250, "altitude_m": 1600},
        Household=lambda prompt: {"state": "Veracruz", "members": 3, "income_month_mxn": 9000},
    )
    respond = api(lambda request: httpx.Response(200, json=PREDICTED))
    adapter = domains.coffee.adapter()

    two = predict(generator, adapter, respond, "A bag?", shown=["review", "offer"])
    one = predict(generator, adapter, respond, "A household?", shown=["households"])
    none = predict(generator, adapter, respond, "Anything?", shown=[])

    assert two.model == "offer" and "- zones" not in generator.asked("ModelChoice")[0]
    assert two.request == {"shop": "almanegra", "bag_grams": 250.0}  # 1,600 m never stated
    assert one.model == "households" and len(generator.asked("ModelChoice")) == 1
    assert one.request["income_month_mxn"] == 9000.0  # required: kept, stated or not
    assert none.error == "No model the API serves answers it" and none.response is None


def test_a_prediction_service_that_fails_is_said_rather_than_raised() -> None:
    generator = Scripted(
        ModelChoice=lambda prompt: {"model": "review"},
        Lot=lambda prompt: {"country": "Ethiopia"},
    )

    answer = predict(generator, domains.coffee.adapter(), api(lambda r: httpx.Response(500)), "?")

    assert answer.response is None
    assert answer.error is not None and "500" in answer.error


# --- The workflow --------------------------------------------------------------------------


def test_a_data_question_is_answered_from_the_tables_and_cites_them(
    session: duckdb.DuckDBPyConnection,
) -> None:
    generator = Scripted(
        RouteReply=lambda p: {"route": "data"},
        SqlReply=lambda p: {
            "sql": "SELECT state FROM clean.mexico_production ORDER BY production_t DESC LIMIT 1"
        },
        AnswerReply=lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    )

    reply = agent(generator, session).ask("Which state produced the most?")

    assert (reply.route, reply.text, reply.verified) == ("data", "Chiapas [sql].", True)
    assert reply.sources == ["[sql] the tables, with the query below"]
    assert reply.sql is not None and reply.sql.result is not None
    assert "[sql] Query result" in generator.asked("AnswerReply")[0]
    assert not generator.asked("PlanReply")  # one tool: no plan to make


def test_a_mixed_question_is_split_and_each_part_goes_to_its_tool(
    session: duckdb.DuckDBPyConnection,
) -> None:
    generator = Scripted(
        RouteReply=lambda p: {"route": "mixed"},
        PlanReply=lambda p: {
            "parts": [
                {"question": "Top state?", "tool": "data"},
                {"question": "Why?", "tool": "knowledge"},
                {"question": "And why altitude?", "tool": "knowledge"},  # one tool, two parts
            ]
        },
        SqlReply=lambda p: {"sql": "SELECT max(production_t) FROM clean.mexico_production"},
        AnswerReply=lambda p: {
            "text": "391,690.56 t [sql]; cooler temperatures [c1].",
            "citations": ["sql", "c1"],
        },
    )

    reply = agent(generator, session).ask("Top state, and why altitude?")

    assert reply.verified
    assert reply.sources == [
        "[sql] the tables, with the query below",
        '[c1] FAO, "Arabica coffee manual" (2005), page 12',
    ]
    assert "Question: Top state?" in generator.asked("SqlReply")[0]
    assert reply.prediction is None
    # The planner is told what each model predicts, as the second opinion is.
    assert "  - review: a cup score" in generator.asked("PlanReply")[0]


def test_a_prediction_question_cites_the_model_and_its_version(
    session: duckdb.DuckDBPyConnection,
) -> None:
    generator = Scripted(
        RouteReply=lambda p: {"route": "prediction"},
        ModelChoice=lambda p: {"model": "offer"},
        Offer=lambda p: {"shop": "almanegra", "bag_grams": 250},
        AnswerReply=lambda p: {"text": "1324.93 MXN/kg [prediction].", "citations": []},
    )

    reply = agent(generator, session).ask("A 250 g bag at Almanegra?")

    assert reply.verified
    assert reply.sources == ["[prediction] the offer model, v5"]
    assert "price_mxn_per_kg = 1324.93" in generator.asked("AnswerReply")[0]
    # A model is offered with what it predicts, since its name need not say.
    assert "- review (predicts total_cup_points): " in generator.asked("ModelChoice")[0]
    # The fields and their vocabulary are in the prompt: Ollama never shows the schema.
    described = generator.asked("Offer")[0]
    assert "- bag_grams (number, required): The bag's size, grams" in described
    assert "- processing_method (string): washed, natural, honey, semi_washed or other" in described
    assert "- observed_on (date): Defaults to today (UTC)." in described


def test_a_plan_that_names_no_tool_asks_the_tables_and_the_documents(
    session: duckdb.DuckDBPyConnection,
) -> None:
    generator = Scripted(
        RouteReply=lambda p: {"route": "mixed"},
        PlanReply=lambda p: {"parts": [{"question": " ", "tool": "data"}]},
        SqlReply=lambda p: {"sql": "SELECT 1 AS one"},
        AnswerReply=lambda p: {"text": "Yes [c1].", "citations": ["c1"]},
    )

    reply = agent(generator, session).ask("Something?")

    assert reply.sql is not None
    assert "[c1]" in generator.asked("AnswerReply")[0]


def test_a_plan_that_hands_the_question_whole_to_two_tools_is_asked_once_more(
    session: duckdb.DuckDBPyConnection,
) -> None:
    whole = "What does the standard measure, and which state grew the most?"
    unsplit = {"parts": [{"question": whole, "tool": "knowledge"},
                         {"question": f" {whole.upper()}", "tool": "data"}]}  # fmt: skip
    split = {"parts": [{"question": "What does the standard measure?", "tool": "knowledge"},
                       {"question": "Which state grew the most?", "tool": "data"}]}  # fmt: skip
    generator = Scripted(
        RouteReply=lambda p: {"route": "mixed"},
        PlanReply=iter([unsplit, split]),
        SqlReply=lambda p: {"sql": TOP},
        AnswerReply=lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    )

    agent(generator, session).ask(whole)

    plans = generator.asked("PlanReply")
    assert len(plans) == 2 and "gave two tools the same question" in plans[1]
    assert "Question: Which state grew the most?" in generator.asked("SqlReply")[0]


def test_an_unsupported_answer_is_written_again_and_the_better_one_kept(
    session: duckdb.DuckDBPyConnection,
) -> None:
    fixed = Scripted(
        RouteReply=lambda p: {"route": "knowledge"},
        AnswerReply=[
            {"text": "Acidity rises by 40% [c1].", "citations": ["c1"]},  # a figure it made up
            {"text": "Acidity develops at altitude [c1].", "citations": ["c1"]},
        ],
    )
    reply = agent(fixed, session).ask("Why altitude?")

    assert (reply.text, reply.verified) == ("Acidity develops at altitude [c1].", True)
    rewrite = fixed.asked("AnswerReply")[1]
    assert "The figure 40% is not in the evidence" in rewrite

    worse = Scripted(
        RouteReply=lambda p: {"route": "knowledge"},
        AnswerReply=[
            {"text": "Acidity rises by 40% [c1].", "citations": ["c1"]},
            {"text": "Acidity rises by 40% and 50% [c7].", "citations": ["c7"]},
        ],
    )
    kept = agent(worse, session).ask("Why altitude?")

    assert kept.text == "Acidity rises by 40% [c1]."  # the rewrite broke more than it fixed
    assert kept.problems == ["The figure 40% is not in the evidence; use only figures it gives."]


def test_every_answer_is_one_trace_that_names_its_prompts(
    session: duckdb.DuckDBPyConnection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("agent")  # not whichever an earlier test left active
    generator = Scripted(
        RouteReply=lambda p: {"route": "knowledge"},
        AnswerReply=lambda p: {"text": "Acidity develops [c1].", "citations": ["c1"]},
    )

    agent(generator, session).ask("Why altitude?")
    mlflow.flush_trace_async_logging()  # traces are written in the background

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())  # type: ignore[arg-type]
    assert trace is not None
    assert trace.info.tags["prompt.answer"] == "prompts:/mlops-agent-answer/1"
    names = [span.name for span in trace.data.spans]
    assert names[0] == "agent"
    assert {"route", "knowledge", "answer", "RouteReply", "AnswerReply"} <= set(names)


# --- The prompt registry -------------------------------------------------------------------


def test_a_prompt_is_registered_when_its_content_changes_and_never_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    first = register_prompts(uri)
    again = register_prompts(uri)
    monkeypatch.setitem(registry.PROMPTS, "router", ("Route {question} anew.", None))
    changed = register_prompts(uri)

    assert set(first) == set(PROMPTS)
    assert first == again
    assert first["router"] == f"prompts:/{PREFIX}router/1"
    assert changed["router"] == f"prompts:/{PREFIX}router/2"
    assert mlflow.genai.load_prompt(changed["router"]).template == "Route {{question}} anew."


# --- The command ---------------------------------------------------------------------------

Script = Callable[[dict[str, Any]], dict[str, Any]]  # a reply shape's properties -> a reply


@pytest.fixture
def stood_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[Script], None]:
    """Everything a real run needs, stood in for: a corpus and its index, the prediction
    API, and - once given a script - Ollama answering by the shape it is asked for."""
    data = tmp_path / "data" / "coffee"
    chunks = pl.DataFrame([PASSAGE | {"chunk": 1, "characters": 60, "topics": ["cultivation"],
                                      "topics_basis": "terms"}], schema=CHUNKS_COLUMNS)  # fmt: skip
    write_table(chunks, data / "clean" / CHUNKS_TABLE, {})
    write_table(pl.DataFrame({"document_id": ["fao"], "title": ["Manual"], "publisher": ["FAO"],
                              "year": [2005]}), data / "clean" / DOCUMENTS_TABLE, {})  # fmt: skip
    client = QdrantClient(":memory:")
    partition = latest_partition(data / "clean" / CHUNKS_TABLE)
    assert partition is not None
    vectors = np.ones((1, 4), dtype=np.float32) / 2
    build_index(client, "coffee", chunks, embedding_table(chunks, vectors),
                {"chunks_partition": partition.name, "chunks_digest": chunks_digest(chunks),
                 "embedding_model": "e@1"}, "s")  # fmt: skip
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "_qdrant", lambda url: client)
    monkeypatch.setattr(
        cli, "_api_client", lambda url: api(lambda r: httpx.Response(200, json=PREDICTED))
    )
    monkeypatch.chdir(tmp_path)
    tags = json.loads((OLLAMA / "tags.json").read_text(encoding="utf-8"))

    def with_ollama(script: Script) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json=tags)
            if request.url.path == "/api/embed":  # one vector per text, all alike
                texts = json.loads(request.content)["input"]
                return httpx.Response(200, json={"embeddings": [[0.5] * 4 for _ in texts]})
            shape = json.loads(request.content)["format"]["properties"]
            reply = script(shape)
            if "predicts" in shape and "predicts" not in reply:
                reply = NOTHING_TO_ADD
            if "answered" in shape:
                reply = {"answered": True} | reply
            return httpx.Response(200, json={"message": {"content": json.dumps(reply)}})

        @contextmanager
        def served(url: str) -> Iterator[httpx.Client]:
            with ollama_client(url, httpx.MockTransport(handler)) as c:
                yield c

        monkeypatch.setattr(llm, "ollama_client", served)

    return with_ollama


@pytest.fixture
def ask(stood_in: Callable[[Script], None]) -> Callable[[str, Script], Any]:
    """`mlops agent ask`, in the stand-in environment."""

    def run(question: str, script: Script) -> Any:
        stood_in(script)
        return CliRunner().invoke(cli.app, ["agent", "ask", question])

    return run


def test_ask_answers_with_its_sources_and_the_trace_it_left(
    ask: Callable[[str, Script], Any],
) -> None:
    def script(shape: dict[str, Any]) -> dict[str, Any]:
        if "steps" in shape:  # coffee plans its questions
            ask = "Why does altitude matter?"
            return {"in_scope": True, "steps": [{"id": "s1", "tool": "knowledge", "ask": ask,
                                                 "uses": []}]}  # fmt: skip
        return {"text": "Altitude delays ripening [c1].", "citations": ["c1"]}

    result = ask("Why does altitude matter?", script)

    assert result.exit_code == 0, result.output
    assert "Altitude delays ripening [c1]." in result.output
    assert '[c1] FAO, "Manual" (2005), page 12' in result.output
    assert "route: knowledge | trace: tr-" in result.output


def test_ask_shows_the_query_the_item_and_what_it_could_not_verify(
    ask: Callable[[str, Script], Any],
) -> None:
    def script(shape: dict[str, Any]) -> dict[str, Any]:
        if "steps" in shape:
            steps = [("s1", "How many chunks?", "data"), ("s2", "Price?", "prediction")]
            return {"in_scope": True, "steps": [{"id": i, "tool": tool, "ask": q, "uses": []}
                                                for i, q, tool in steps]}  # fmt: skip
        if "holds" in shape:
            return {"holds": True, "instead": ""}
        if "sql" in shape:
            return {"sql": "SELECT count(*) AS n FROM clean.document_chunks"}
        if "model" in shape:
            return {"model": "offer"}
        if "shop" in shape:
            return {"shop": "almanegra", "bag_grams": 250}
        return {"text": "There are 99 [sql].", "citations": ["sql"]}  # 99 is made up

    result = ask("How many, and what price?", script)

    assert result.exit_code == 0, result.output
    assert "sql: SELECT count(*) AS n FROM clean.document_chunks" in result.output
    assert "prediction (offer): {'shop': 'almanegra', 'bag_grams': 250.0}" in result.output
    assert "unverified: The figure 99 is not in the evidence" in result.output


def test_with_no_evidence_it_says_so_without_writing_an_answer(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """A prediction the service refused, and nothing else: no answer is written - a model
    handed nothing writes a figure anyway - and the reply says what each tool found."""
    generator = Scripted(
        RouteReply=lambda p: {"route": "prediction"},
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Ethiopia"},
    )

    reply = agent(generator, session, lambda r: httpx.Response(503)).ask("Score?")

    assert generator.asked("AnswerReply") == []
    assert not reply.answered and reply.verified and reply.sources == []
    assert reply.text.startswith("I found no answer")
    assert "The models: no prediction (The prediction service failed" in reply.text


def test_a_query_that_finds_nothing_is_checked_against_the_data_once(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """The value it filtered on is not there; the rewrite is shown the values that are,
    and when it finds nothing too, the documents are what is left."""
    empty = "SELECT production_t FROM clean.mexico_production WHERE state = 'Chiapaz'"
    generator = Scripted(
        RouteReply=lambda p: {"route": "mixed"},
        PlanReply=lambda p: {
            "parts": [
                {"question": "Chiapas?", "tool": "data"},
                {"question": "Why?", "tool": "knowledge"},
            ]
        },
        SqlReply=[{"sql": empty}, {"sql": empty}],
        AnswerReply=lambda p: {"text": "Altitude delays ripening [c1].", "citations": ["c1"]},
    )

    reply = agent(generator, session).ask("How much did Chiapas grow, and why?")

    rewrite = generator.asked("SqlReply")[1]
    assert "found nothing" in rewrite
    assert "state = 'Chiapaz' matches no row of clean.mexico_production" in rewrite
    assert "its values include: Chiapas, Puebla" in rewrite
    assert "The tables have nothing for it" in generator.asked("AnswerReply")[0]
    assert "[sql] Query result" not in generator.asked("AnswerReply")[0]
    assert reply.answered and reply.sql is not None and reply.sql.empty


def test_a_passage_far_from_the_question_is_not_evidence(
    session: duckdb.DuckDBPyConnection,
) -> None:
    generator = Scripted(RouteReply=lambda p: {"route": "knowledge"})
    far = PASSAGE | {"score": 0.2}
    asked = Agent(
        generator, domains.coffee.adapter(), session, "",
        {"subject": "coffee", "tables": "", "models": "  - review: a cup score", "topics": ""},
        lambda question, k: [far], api(lambda r: httpx.Response(500)), {}, {},
    ).ask("Who won the 2022 World Cup?")  # fmt: skip

    assert not asked.answered and asked.passages == []
    assert "The documents: no passage is close enough" in asked.text


def test_a_rewrite_may_fix_a_spelling_but_never_swap_in_another_value(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """Shown the states there are, the rewrite of a query for one that is not there may
    not answer about another: the empty result stands, and the gate says so."""
    missing = "SELECT production_t FROM clean.mexico_production WHERE state = 'Jalisco'"
    swapped = "SELECT production_t FROM clean.mexico_production WHERE state = 'Puebla'"
    misspelt = "SELECT production_t FROM clean.mexico_production WHERE state = 'CHIAPÁS'"
    fixed = "SELECT production_t FROM clean.mexico_production WHERE state = 'Chiapas'"

    kept = write_sql(Scripted(SqlReply=[{"sql": missing}, {"sql": swapped}]), session, "", "?")
    respelt = write_sql(Scripted(SqlReply=[{"sql": misspelt}, {"sql": fixed}]), session, "", "?")

    assert kept.sql == missing and kept.empty
    assert respelt.sql == fixed and not respelt.empty
    assert same_values(
        "SELECT 1 FROM t WHERE shop IN ('Café')", "SELECT 1 FROM t WHERE shop = 'cafe'"
    )
    assert not same_values(
        "SELECT 1 FROM t WHERE shop = 'starbucks'", "SELECT 1 FROM t WHERE shop = 'buna'"
    )


def test_a_query_that_reads_no_table_is_not_an_answer(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """Asked who won a World Cup, the model wrote SELECT 'Brazil': its own figure."""
    constant = "SELECT 'Brazil' AS country"
    generator = Scripted(SqlReply=[{"sql": constant}] * 3)

    answer = write_sql(generator, session, "", "Who won?")

    assert answer.result is None and answer.error == NO_TABLE
    assert NO_TABLE in generator.asked("SqlReply")[1]
    rewritten = write_sql(
        Scripted(SqlReply=[{"sql": "SELECT state FROM clean.mexico_production WHERE state = 'X'"},
                           {"sql": constant}]),
        session, "", "?",
    )  # fmt: skip
    assert rewritten.sql.endswith("'X'")  # a rewrite that reads no table is not kept


def test_a_query_that_ignores_a_guard_is_written_again_once(
    session: duckdb.DuckDBPyConnection,
) -> None:
    guard = SqlGuard(table="clean.mexico_production", requires="year", hint="Filter a year.")
    unguarded = "SELECT state FROM clean.mexico_production ORDER BY production_t DESC LIMIT 1"
    generator = Scripted(SqlReply=[{"sql": unguarded}, {"sql": unguarded + " -- year"}])

    answer = write_sql(generator, session, "", "Top state?", guards=[guard])

    assert "It ran, but: Filter a year." in generator.asked("SqlReply")[1]
    assert answer.attempts == 2 and answer.sql.endswith("-- year")
    assert write_sql(Scripted(SqlReply=[{"sql": TOP}]), session, "", "Top?").attempts == 1


def test_a_rewrite_that_fails_keeps_the_query_that_ran(
    session: duckdb.DuckDBPyConnection,
) -> None:
    empty = "SELECT state FROM clean.mexico_production WHERE state IN ('Nowhere')"
    broken = "SELECT nonsense FROM clean.mexico_production WHERE state IN ('Nowhere')"
    generator = Scripted(SqlReply=[{"sql": empty}, {"sql": broken}])

    answer = write_sql(generator, session, "", "Nowhere?")

    assert answer.sql == empty and answer.empty and answer.attempts == 2
    assert "state = 'Nowhere' matches no row" in generator.asked("SqlReply")[1]


def test_only_a_value_the_data_lacks_is_reported(session: duckdb.DuckDBPyConnection) -> None:
    """A value that is there (the query found nothing for another reason), and a column
    the table does not have (a name the query gave it), are not what went wrong."""
    names = views(session)
    there = "SELECT state FROM clean.mexico_production WHERE state = 'Chiapas' AND production_t < 0"
    renamed = (
        "WITH t AS (SELECT state AS s FROM clean.mexico_production) SELECT s FROM t WHERE s = 'X'"
    )

    assert absent_values(session, there, names) == []
    assert absent_values(session, renamed, names) == []
    # A pattern is matched as a pattern: '%iapa%' is there, '%Mexico City%' is not.
    pattern = "SELECT count(*) FROM clean.mexico_production WHERE state ILIKE '{}'"
    assert absent_values(session, pattern.format("%iapa%"), names) == []
    assert absent_values(session, pattern.format("%Mexico City%"), names) == [
        "state ILIKE '%Mexico City%' matches no row of clean.mexico_production; "
        "its values include: Chiapas, Puebla"
    ]


def test_a_count_of_zero_goes_back_only_for_a_value_the_data_lacks(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """Zero is an answer when the filter names what is there; a filter on something the
    data names otherwise counts zero of anything, and is checked like nothing found."""
    count = "SELECT count(*) AS n FROM clean.mexico_production"
    lacking, fixed = f"{count} WHERE state LIKE '%Mexico City%'", count
    there = f"{count} WHERE state = 'Chiapas' AND production_t < 0"
    asked = Scripted(SqlReply=[{"sql": lacking}, {"sql": fixed}])

    rewritten = write_sql(asked, session, "", "How many?")
    kept = write_sql(Scripted(SqlReply=[{"sql": there}]), session, "", "How many in Chiapas?")

    assert "or a count of zero" in asked.asked("SqlReply")[1]
    assert rewritten.sql == fixed and rewritten.attempts == 2
    assert kept.sql == there and kept.attempts == 1 and kept.result is not None
    assert kept.result.rows == [(0,)]
    assert counted_nothing(QueryResult("", ["n", "sum"], [(0, None)], False))
    assert not counted_nothing(QueryResult("", ["above"], [(False,)], False))  # an answer
    assert not counted_nothing(QueryResult("", ["n"], [(0,), (0,)], False))


def test_a_filter_the_question_never_names_is_reported_when_nothing_is_found(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """Asked for the roasters' Gesha coffees, the model also filtered on Ethiopia."""
    sql = "SELECT 1 FROM t WHERE variety ILIKE '%gesha%' AND country = 'Ethiopia' AND x = 'ab'"

    assert unasked_values(sql, "The median price of the Gésha coffees?") == [
        "country = 'Ethiopia': the question does not name it"  # 'ab': too short to tell
    ]
    unasked = "SELECT production_t FROM clean.mexico_production WHERE state = 'Chiapas' AND 1 = 0"
    asked = Scripted(SqlReply=[{"sql": unasked}, {"sql": unasked}])
    write_sql(asked, session, "", "How much was grown?")
    assert "state = 'Chiapas': the question does not name it" in asked.asked("SqlReply")[1]


def test_queries_vote_and_a_tie_keeps_the_usual_one(session: duckdb.DuckDBPyConnection) -> None:
    """The usual query and two sampled ones: two agree, the usual one is outvoted. Alone
    against one, or when no query found anything, the usual one stands."""
    bottom = "SELECT state FROM clean.mexico_production ORDER BY production_t LIMIT 1"
    voters = [Scripted(SqlReply=[{"sql": TOP}]), Scripted(SqlReply=[{"sql": f"{TOP};"}])]

    outvoted = write_sql(Scripted(SqlReply=[{"sql": bottom}]), session, "", "Top?", voters=voters)
    one = [Scripted(SqlReply=[{"sql": TOP}])]
    tied = write_sql(Scripted(SqlReply=[{"sql": bottom}]), session, "", "Top?", voters=one)

    assert outvoted.result is not None and outvoted.result.rows == [("Chiapas",)]
    assert tied.sql == bottom
    nothing = SqlAnswer("SELECT 1 WHERE false", QueryResult("", ["x"], [], False), None, 1)
    failed = SqlAnswer("SELECT nonsense", None, "Binder Error", 3)
    assert voted([nothing, failed]) is nothing
    # Rows in any order, numbers to four significant figures, are one result.
    ran = SqlAnswer("q1", QueryResult("", ["a"], [(1.00001,), (2,)], False), None, 1)
    same = SqlAnswer("q2", QueryResult("", ["b"], [(2.0,), (1.0,)], False), None, 1)
    other = SqlAnswer("q3", QueryResult("", ["a"], [(3,)], False), None, 1)
    assert voted([other, ran, same]) is ran


def test_the_agent_lets_its_voters_write_the_query(session: duckdb.DuckDBPyConnection) -> None:
    voter = Scripted(SqlReply=lambda p: {"sql": TOP})
    generator = Scripted(
        RouteReply=lambda p: {"route": "data"},
        SqlReply=lambda p: {"sql": TOP},
        AnswerReply=lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    )

    agent(generator, session, voters=[voter]).ask("Which state grew the most?")

    assert "Question: Which state grew the most?" in voter.asked("SqlReply")[0]


def test_a_result_past_the_row_cap_goes_back_once(session: duckdb.DuckDBPyConnection) -> None:
    """The rows asked about may be past the cut: the writer is told, once."""
    every = "SELECT state FROM clean.mexico_production"
    generator = Scripted(SqlReply=[{"sql": every}, {"sql": TOP}])

    answer = write_sql(generator, session, "", "Which state grew the most?", max_rows=1)

    assert "matched more than 1 rows" in generator.asked("SqlReply")[1]
    assert answer.sql == TOP and answer.attempts == 2
    assert write_sql(Scripted(SqlReply=[{"sql": TOP}]), session, "", "?", max_rows=1).attempts == 1


def test_an_error_goes_back_with_what_it_means_and_a_sentence_is_not_a_value(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """A query that selects its own sentence is refused before it runs; an error whose words
    do not say how to mend the query goes back with a hint."""
    prose = (
        "SELECT 'The standard measures ten attributes of the cup' AS what, max(production_t) "
        "FROM clean.mexico_production"
    )
    listed = "SELECT state FROM clean.mexico_production WHERE unnest([state]) = 'Chiapas'"
    generator = Scripted(SqlReply=[{"sql": prose}, {"sql": listed}, {"sql": TOP}])

    answer = write_sql(generator, session, "", "What does it measure, and the top state?")

    repairs = generator.asked("SqlReply")[1:]
    assert PROSE in repairs[0]
    assert "UNNEST not supported here" in repairs[1] and "list_contains" in repairs[1]
    assert answer.sql == TOP and answer.attempts == 3
    assert writes_prose("SELECT 'It''s what the cup holds, in short' AS x FROM t")
    assert not writes_prose(
        "SELECT n FROM t WHERE kind = 'Cafeterías, fuentes de sodas y neverías'"
    )
    assert repair_hint("Binder Error: something else") == ""


def test_when_the_tables_find_nothing_the_reply_says_what_the_query_did(
    session: duckdb.DuckDBPyConnection,
) -> None:
    empty = "SELECT production_t FROM clean.mexico_production WHERE state = 'Jalisco'"
    found_nothing = Scripted(RouteReply=lambda p: {"route": "data"}, SqlReply=[{"sql": empty}] * 2)
    failed = Scripted(
        RouteReply=lambda p: {"route": "data"}, SqlReply=lambda p: {"sql": "SELECT x FROM nowhere"}
    )

    nothing = agent(found_nothing, session).ask("How much did Jalisco grow?")
    broken = agent(failed, session).ask("How much did Jalisco grow?")

    assert not nothing.answered and found_nothing.asked("AnswerReply") == []
    assert "The tables: the query ran and found nothing." in nothing.text
    assert not broken.answered and "The tables: the query failed (" in broken.text


def test_the_second_opinion_adds_a_tool_and_never_takes_one_away(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """A lot the question describes and no figure asked of the tables, routed to the
    tables: it goes to the model instead. A figure asked of a question routed to the
    documents: the tables are added."""
    described = Scripted(
        RouteReply=lambda p: {"route": "data"},
        NeedsReply=lambda p: {"predicts": True, "figures": False},
        SqlReply=[{"sql": TOP}],
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya"},
        AnswerReply=lambda p: {"text": "84.4 [prediction].", "citations": ["prediction"]},
    )
    counted = Scripted(
        RouteReply=lambda p: {"route": "knowledge"},
        NeedsReply=lambda p: {"predicts": False, "figures": True},
        SqlReply=[{"sql": TOP}],
        AnswerReply=lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    )

    both = Scripted(
        RouteReply=lambda p: {"route": "data"},
        NeedsReply=lambda p: {"predicts": True, "figures": True},
        SqlReply=[{"sql": TOP}],
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya"},
        AnswerReply=lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    )

    first = agent(described, session).ask("What would a Kenyan lot score?")
    second = agent(counted, session).ask("Which state grows most, and why?")
    # The prediction service is down: the tables still answer, and the answer is told.
    third = agent(both, session, lambda r: httpx.Response(503)).ask("Top state, and a score?")

    assert first.route == "prediction" and first.prediction is not None and first.sql is None
    assert second.route == "mixed" and second.sql is not None and second.sql.sql == TOP
    assert third.route == "mixed" and third.answered and third.prediction is not None
    assert "No prediction: " in both.asked("AnswerReply")[0]


def test_a_domain_without_documents_sends_what_was_meant_for_them_to_the_tables(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """A business has no library: a route to one could only end in no answer, so the
    question - or the part of it meant for the documents - is asked of the tables."""
    shapes: dict[str, Any] = {
        "SqlReply": lambda p: {"sql": TOP},
        "AnswerReply": lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    }
    routed = Scripted(RouteReply=lambda p: {"route": "knowledge"}, **shapes)
    split = Scripted(
        RouteReply=lambda p: {"route": "mixed"},
        PlanReply=lambda p: {"parts": [{"question": "Why does it grow?", "tool": "knowledge"}]},
        **shapes,
    )
    both = Scripted(
        RouteReply=lambda p: {"route": "mixed"},
        PlanReply=lambda p: {
            "parts": [
                {"question": "Which state grows most?", "tool": "data"},
                {"question": "Why there?", "tool": "knowledge"},
            ]
        },
        **shapes,
    )

    first = agent(routed, session, library=False).ask("Which state grows most?")
    second = agent(split, session, library=False).ask("Why does it grow?")
    third = agent(both, session, library=False).ask("Which state grows most, and why?")

    assert first.route == "data" and first.sql is not None and first.passages == []
    assert second.sql is not None and "Why does it grow?" in split.asked("SqlReply")[0]
    assert third.sql is not None and "Which state grows most?" in both.asked("SqlReply")[0]
    assert second.passages == [] and third.passages == []


def test_the_closest_studies_are_shown_to_the_router_the_second_opinion_and_the_plan(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """An answer already computed is in front of every step that chooses a tool."""
    card = StudyCard("analysis.harvests", "what a harvest would bring at every price", "")
    studies = StudyFinder([card], lambda texts: np.ones((len(texts), 2)), shown=1)
    generator = Scripted(
        RouteReply=lambda p: {"route": "mixed"},
        PlanReply=lambda p: {"parts": [{"question": "Which state?", "tool": "data"}]},
        SqlReply=lambda p: {"sql": TOP},
        AnswerReply=lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    )

    agent(generator, session, studies=studies).ask("What would a harvest bring?")

    listed = "analysis.harvests: what a harvest would bring at every price"
    for shape in ("RouteReply", "NeedsReply", "PlanReply"):
        assert listed in generator.asked(shape)[0], shape

    # Shown the studies, a router's "data" stands: the second opinion adds a prediction,
    # and no longer takes the tables' place.
    what_if = Scripted(
        RouteReply=lambda p: {"route": "data"},
        NeedsReply=lambda p: {"predicts": True, "figures": False},
        SqlReply=lambda p: {"sql": TOP},
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya"},
        AnswerReply=lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    )
    both = agent(what_if, session, studies=studies).ask("What would a harvest bring?")
    assert both.route == "mixed" and both.sql is not None and both.prediction is not None


def test_a_result_that_holds_something_else_is_no_answer(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """Asked for one thing, the tables returned another: not evidence, and the reply says
    what they hold. A result that holds what was asked is kept; and a mixed question keeps
    its prediction when its query misses."""
    shapes: dict[str, Any] = {
        "SqlReply": lambda p: {"sql": TOP},
        "AnswerReply": lambda p: {"text": "Chiapas [sql].", "citations": ["sql"]},
    }
    elsewhere = Scripted(
        RouteReply=lambda p: {"route": "data"},
        HoldsReply=lambda p: {"holds": False, "instead": "the states' own harvests"},
        **shapes,
    )
    kept = Scripted(
        RouteReply=lambda p: {"route": "data"},
        HoldsReply=lambda p: {"holds": True, "instead": ""},
        **shapes,
    )
    mixed = Scripted(
        RouteReply=lambda p: {"route": "data"},
        NeedsReply=lambda p: {"predicts": True, "figures": True},
        HoldsReply=lambda p: {"holds": False, "instead": " "},
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya"},
        SqlReply=lambda p: {"sql": TOP},
        AnswerReply=lambda p: {"text": "84.4 [prediction].", "citations": ["prediction"]},
    )

    refused = agent(elsewhere, session, check_result=True).ask("What do importers pay?")
    answered = agent(kept, session, check_result=True).ask("Which state grows most?")
    partly = agent(mixed, session, check_result=True).ask("What would a Kenyan lot score?")

    assert not refused.answered and refused.sql is not None and refused.sql.result is None
    assert "it holds the states' own harvests." in refused.text
    checked = elsewhere.asked("HoldsReply")[0]
    assert "Question: What do importers pay?" in checked
    assert "## `clean.mexico_production` — coffee grown" in checked  # the table it read
    assert answered.answered and answered.sql is not None and answered.sql.result is not None
    assert partly.answered and partly.prediction is not None
    assert "it holds something else" in mixed.asked("AnswerReply")[0]


# --- The planner ------------------------------------------------------------------------


def planned(*steps: tuple[str, str, str, list[str]], in_scope: bool = True) -> dict[str, Any]:
    return {
        "in_scope": in_scope,
        "steps": [
            {"id": i, "tool": tool, "ask": ask, "uses": uses} for i, tool, ask, uses in steps
        ],
    }


def test_a_question_about_something_else_is_declined_before_any_tool(
    session: duckdb.DuckDBPyConnection,
) -> None:
    generator = Scripted(StepPlan=lambda p: planned(in_scope=False))

    reply = agent(generator, session, planner=True).ask("Who won the 2022 World Cup?")

    assert reply.route == "none" and not reply.answered and reply.steps == ()
    assert reply.text.startswith("That is not something I can answer: I answer questions about")
    assert [name for name, _ in generator.prompts] == ["StepPlan"]
    plan_prompt = generator.asked("StepPlan")[0]
    assert "- data: the domain's tables" in plan_prompt and "- knowledge:" in plan_prompt
    # An ask never names whose records it reads: a name becomes a filter.
    assert "Every source already holds coffee and nothing else" in plan_prompt


def test_a_step_that_needs_another_runs_after_it_and_is_handed_what_it_found(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """The prediction is asked with the altitude the tables found: a number the question
    never states, kept because a step it needs found it."""
    generator = Scripted(
        StepPlan=lambda p: planned(
            ("s1", "data", "At what altitude does the top state grow?", []),
            ("s2", "prediction", "What would a Kenyan lot at the altitude s1 finds score?", ["s1"]),
        ),
        SqlReply=lambda p: {
            "sql": "SELECT 1450.0 AS altitude_m FROM clean.mexico_production LIMIT 1"
        },
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya", "altitude_m": 1450.0},
        AnswerReply=lambda p: {
            "text": "At 1450 m [sql], 1324.93 [prediction].",
            "citations": ["sql", "prediction"],
        },
    )

    reply = agent(generator, session, planner=True).ask(
        "At the top state's altitude, a Kenyan lot?"
    )

    assert reply.route == "mixed" and [s.step.id for s in reply.steps] == ["s1", "s2"]
    names = [name for name, _ in generator.prompts]
    assert names.index("SqlReply") < names.index("Lot")
    described_item = generator.asked("Lot")[0]
    assert "Found by the steps this one uses:\n[s1] At what altitude" in described_item
    assert reply.prediction is not None and reply.prediction.request["altitude_m"] == 1450.0
    assert reply.answered and reply.verified
    assert reply.sources == [
        "[sql] the tables, with the query below",
        "[prediction] the review model, v5",
    ]


def test_steps_that_need_nothing_from_each_other_run_at_once(
    session: duckdb.DuckDBPyConnection,
) -> None:
    threads: set[int] = set()

    def query(prompt: str) -> dict[str, Any]:
        threads.add(threading.get_ident())
        time.sleep(0.05)
        return {"sql": TOP}

    generator = Scripted(
        StepPlan=lambda p: planned(
            ("s1", "data", "Which state grows most?", []),
            ("s2", "data", "Which state grows most, again?", []),
        ),
        SqlReply=query,
        AnswerReply=lambda p: {
            "text": "Chiapas [sql], Chiapas [sql2].",
            "citations": ["sql", "sql2"],
        },
    )

    reply = agent(generator, session, planner=True).ask("Which state grows most, twice?")

    assert len(threads) == 2
    evidence = generator.asked("AnswerReply")[0]
    assert '[sql] Query result for "Which state grows most?"' in evidence
    assert '[sql2] Query result for "Which state grows most, again?"' in evidence
    assert reply.route == "data" and reply.sources == [
        "[sql] the tables, with the query below",
        "[sql2] the tables, with the query below",
    ]


def test_a_step_whose_need_found_nothing_is_not_run_and_the_reply_says_why(
    session: duckdb.DuckDBPyConnection,
) -> None:
    generator = Scripted(
        StepPlan=lambda p: planned(
            ("s1", "data", "What altitude does Iceland grow at?", []),
            ("s2", "prediction", "What would a lot at that altitude score?", ["s1"]),
        ),
        SqlReply=lambda p: {
            "sql": "SELECT state FROM clean.mexico_production WHERE state = 'Iceland'"
        },
    )

    reply = agent(generator, session, planner=True).ask("Iceland's altitude, and a lot there?")

    assert not reply.answered and reply.steps[1].skipped is not None
    assert "s2: not looked up (it needs what s1 found, and that found nothing)." in reply.text
    assert "s1: The tables: the query ran and found nothing." in reply.text
    assert not generator.asked("Lot")


def test_a_plan_that_cannot_run_is_asked_again_then_left_to_the_router(
    session: duckdb.DuckDBPyConnection,
) -> None:
    wrong = planned(("s1", "data", "Which state?", ["s0"]))
    right = planned(("s1", "data", "Which state grows most?", []))
    answer = {"text": "Chiapas [sql].", "citations": ["sql"]}
    twice = Scripted(StepPlan=[wrong, wrong], RouteReply=lambda p: {"route": "data"},
                     SqlReply=lambda p: {"sql": TOP}, AnswerReply=lambda p: answer)  # fmt: skip
    again = Scripted(StepPlan=[wrong, right], SqlReply=lambda p: {"sql": TOP},
                     AnswerReply=lambda p: answer)  # fmt: skip

    routed = agent(twice, session, planner=True).ask("Which state grows most?")
    replanned = agent(again, session, planner=True).ask("Which state grows most?")

    told = "Your plan could not be run:\n- Step s1 uses s0, which is not a step planned before it."
    assert told in twice.asked("StepPlan")[1]
    assert routed.answered and routed.route == "data" and len(twice.asked("RouteReply")) == 1
    assert [s.step.ask for s in routed.steps] == ["Which state grows most?"]
    assert replanned.answered and not again.asked("RouteReply")


def test_two_predictions_are_each_read_from_their_own_step(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """The asker's words hold both items: each step's own words hold one."""
    generator = Scripted(
        StepPlan=lambda p: planned(
            ("s1", "prediction", "What would a Kenyan lot score?", []),
            ("s2", "prediction", "What would an Ethiopian lot score?", []),
        ),
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya" if "Kenyan" in p.split("Question:")[-1] else "Ethiopia"},
        AnswerReply=lambda p: {
            "text": "1324.93 [prediction], 1324.93 [prediction2].",
            "citations": ["prediction", "prediction2"],
        },
    )

    reply = agent(generator, session, planner=True).ask("A Kenyan lot and an Ethiopian one?")

    asked = [prompt.split("Question:")[-1] for prompt in generator.asked("Lot")]
    assert any("Kenyan lot score" in a and "Ethiopian" not in a for a in asked)
    assert any("Ethiopian lot score" in a and "Kenyan" not in a for a in asked)
    assert {s.prediction.request["country"] for s in reply.steps if s.prediction} == {
        "Kenya",
        "Ethiopia",
    }
    assert reply.route == "prediction" and "[prediction2] the review model, v5" in reply.sources


def test_documents_in_a_plan_and_what_each_step_found_when_none_is_evidence(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """Two searches that find the same passage give it once; and when no step finds
    anything usable, the reply says, step by step, what each came back with."""
    found = Scripted(
        StepPlan=lambda p: planned(
            ("s1", "knowledge", "Why does altitude matter?", []),
            ("s2", "knowledge", "How does altitude change the cup?", []),
        ),
        AnswerReply=lambda p: {"text": "It slows ripening [c1].", "citations": ["c1"]},
    )
    far = PASSAGE | {"score": 0.2}
    nothing = Scripted(
        StepPlan=lambda p: planned(
            ("s1", "prediction", "What would a Kenyan lot score?", []),
            ("s2", "knowledge", "Why?", []),
        ),
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Kenya"},
    )
    down = Agent(
        nothing, domains.coffee.adapter(), session, "",
        {"subject": "coffee", "tables": "", "models": "  - review: a cup score", "topics": ""},
        lambda question, k: [far], api(lambda r: httpx.Response(503)), {}, {}, planner=True,
    )  # fmt: skip

    answered = agent(found, session, planner=True).ask("Why does altitude matter?")
    missed = down.ask("A Kenyan lot's score, and why?")

    assert answered.answered and len(answered.passages) == 1 and answered.route == "knowledge"
    assert not missed.answered
    assert "s1: the models: no prediction (The prediction service failed:" in missed.text
    assert "s2: the documents: no passage is close enough to the question." in missed.text


def test_ask_lists_each_step_of_a_plan(
    stood_in: Callable[[Script], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sql = SqlAnswer(TOP, QueryResult(TOP, ["state"], [("Chiapas",)], False), None, 1)
    item = PredictionAnswer("review", {"country": "Kenya"}, PREDICTED, None)
    steps = (
        StepResult(PlannedStep(id="s1", tool="data", ask="Top state?", uses=[]), sql=sql),
        StepResult(PlannedStep(id="s2", tool="prediction", ask="Score?", uses=[]), prediction=item),
        StepResult(PlannedStep(id="s3", tool="data", ask="Again?", uses=["s1"]), skipped="no"),
    )
    reply = Reply("Q?", "mixed", "Chiapas [sql].", ["[sql] the tables, with the query below"],
                  sql, item, [], [], steps=steps)  # fmt: skip
    # One step, or the router's: the query and the item, each on a line of its own.
    single = Reply("Q?", "mixed", "Chiapas [sql].", [], sql, item, [], [], steps=steps[:1])
    replies = iter([reply, single])

    @contextmanager
    def session(*args: Any, **kwargs: Any) -> Iterator[tuple[Any, str]]:
        answer = next(replies)
        yield type("Stub", (), {"ask": staticmethod(lambda question: answer)})(), "stub"

    monkeypatch.setenv(
        "MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'ask-mlflow.db').as_posix()}"
    )
    monkeypatch.setattr(cli, "agent_session", session)
    result = CliRunner().invoke(cli.app, ["agent", "ask", "Q?"])
    one = CliRunner().invoke(cli.app, ["agent", "ask", "Q?"])

    assert one.exit_code == 0, one.output
    assert "\nsql: SELECT state" in one.output and "\nprediction (review): {'country'" in one.output
    assert "step s1" not in one.output

    assert result.exit_code == 0, result.output
    assert "step s1 (data): Top state?\n  sql: SELECT state" in result.output
    assert (
        "step s2 (prediction): Score?\n  prediction (review): {'country': 'Kenya'}" in result.output
    )
    assert "step s3 (data): Again?\n  not run: no" in result.output


def test_the_prediction_api_is_reached_at_the_configured_address() -> None:
    with cli._api_client("http://api.example:8000") as client:
        assert str(client.base_url) == "http://api.example:8000"


def test_evaluate_asks_every_question_and_compares_with_the_last_run(
    stood_in: Callable[[Script], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs of `mlops agent evaluate` on a domain with two questions, whose references
    live in the other sets under the same words; the second run's query is wrong."""
    home = tmp_path / "domain"
    (home / "evals").mkdir(parents=True)
    real = domain_dir("coffee")
    (home / "data_dictionary.md").write_text(
        (real / "data_dictionary.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    lines = {
        "routing_questions.jsonl": [
            {"id": "data-01", "question": "Top state?", "route": "data"},
            {"id": "knowledge-01", "question": "Why altitude?", "route": "knowledge"},
            {"id": "data-02", "question": "Who rents?", "route": "data"},  # not its to answer
        ],
        "sql_questions.jsonl": [
            {"id": "production-01", "question": "Top state?", "sql": TOP},
            {"id": "profile-01", "question": "Who rents?",
             "sql": "SELECT borough FROM clean.borough_profile"},
        ],
        "retrieval_questions.jsonl": [
            {"id": "cultivation-01", "topic": "cultivation", "question": "Why altitude?",
             "answer": "It delays ripening.", "status": "draft", "drafted_by": "m@1",
             "prompt": "p", "drafted_on": "2026-09-24", "source_chunk": "doc-0001",
             "relevant": [{"document_id": "fao", "part": 12, "grade": 2,
                           "excerpt": "Cooler temperatures at altitude delay ripening"}]},
        ],
    }  # fmt: skip
    for name, rows in lines.items():
        text = "".join(json.dumps(row) + "\n" for row in rows)
        (home / "evals" / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(cli, "domain_dir", lambda name: home)
    write_table(pl.DataFrame({"state": ["Chiapas", "Puebla"], "production_t": [391690.56, 1.0]}),
                tmp_path / "data" / "coffee" / "clean" / "mexico_production", {})  # fmt: skip

    def run(sql: str) -> Any:
        def script(shape: dict[str, Any]) -> dict[str, Any]:
            if "steps" in shape:
                steps = [("s1", "Top state?", "data"), ("s2", "Why altitude?", "knowledge")]
                return {"in_scope": True, "steps": [{"id": i, "tool": tool, "ask": q, "uses": []}
                                                    for i, q, tool in steps]}  # fmt: skip
            if "holds" in shape:
                return {"holds": True, "instead": ""}
            if "sql" in shape:
                return {"sql": sql}
            return {"text": "It is so [sql] [c1].", "citations": ["sql", "c1"]}

        stood_in(script)
        return CliRunner().invoke(cli.app, ["agent", "evaluate"])

    first = run(TOP)
    second = run(TOP.replace("DESC", "ASC"))

    assert first.exit_code == 0, first.output
    assert "1 questions read tables the agent is not shown: left out" in first.output
    assert "100% correct, 100% verified, routing 0%" in first.output
    assert "sql: 100% of the questions it applies to" in first.output
    assert "passage: 100% of the questions it applies to" in first.output
    assert "vs the previous run" not in first.output
    assert second.exit_code == 0, second.output
    assert "x data-01: route mixed; wrong query result" in second.output
    assert "vs the previous run: -50% correct" in second.output
    answers = tmp_path / "data" / "coffee" / "evaluations" / "agent_answers"
    assert len(list(answers.glob("built_at=*"))) == 2


# --- MCP -----------------------------------------------------------------------------------


def mcp_server(
    session: duckdb.DuckDBPyConnection,
    respond: Callable[[httpx.Request], httpx.Response] | None = None,
    areas: Any = None,
) -> Any:
    from mlops_core.agent.mcp_server import build_server

    return build_server(
        domains.coffee.adapter(),
        session,
        "## `clean.mexico_production` — coffee grown",
        lambda question, k: [PASSAGE] * k,
        api(respond or (lambda request: httpx.Response(200, json=PREDICTED))),
        lambda passage: f"FAO, page {passage['part']}",
        areas,
    )


def test_the_mcp_server_offers_the_agents_tools_all_read_only(
    session: duckdb.DuckDBPyConnection,
) -> None:
    server = mcp_server(session)

    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    resource = asyncio.run(server.read_resource("dictionary://tables"))

    assert set(tools) == {
        "query_tables",
        "predict_review",
        "predict_offer",
        "predict_green_price",
        *NEW_PREDICTIONS,
        "search_documents",
        "draw",
        "explore_segment",
        "map_layer",
    }
    assert all(t.annotations is not None and t.annotations.read_only_hint for t in tools.values())
    # A prediction tool's input is the model's own request body, descriptions included.
    offer = json.dumps(tools["predict_offer"].input_schema)
    assert "The roaster, as the catalogues name it" in offer
    assert tools["predict_offer"].description.startswith("Predict the price per kilogram")
    assert "`clean.mexico_production`" in next(iter(resource)).content


def test_mcp_offers_the_sources_closest_to_a_question_and_the_agent_itself(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """A domain with no documents has nothing to search; the catalogue's closest studies
    and models, and the agent's own answer with every step, are tools of their own."""
    from mlops_core.agent.mcp_server import build_server, replied, source_finder
    from mlops_core.agent.model_cards import ModelCard, ModelFinder

    card = StudyCard("analysis.scenarios", "a day with every price moved", "", ("change_pct",))
    studies = StudyFinder([card], lambda texts: np.ones((len(texts), 2)), shown=1)
    models = ModelFinder([ModelCard("demand", "Tickets an hour.", ["weekday", "hour"])], None)  # type: ignore[arg-type]
    find = source_finder(studies, models)
    sql = SqlAnswer(TOP, QueryResult(TOP, ["state"], [("Chiapas",)], False), None, 1)
    step = StepResult(PlannedStep(id="s1", tool="data", ask="Top state?", uses=[]), sql=sql)
    scored = PredictionAnswer("review", {"country": "Kenya"}, PREDICTED, None)
    later = StepResult(
        PlannedStep(id="s2", tool="prediction", ask="Score?", uses=["s1"]), prediction=scored
    )
    reply = Reply("Q?", "mixed", "Chiapas [sql].", ["[sql] the tables"], sql, None, [], [],
                  steps=(step, later))  # fmt: skip
    shop = load_adapter("coffee/cafe_de_barrio")
    server = build_server(shop, session, "", lambda q, k: [], api(lambda r: httpx.Response(200)),
                          lambda p: "", finder=find, ask=lambda question: reply)  # fmt: skip

    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    sources = asyncio.run(server.call_tool("find_sources", {"question": "What if prices rose?"}))
    answer = asyncio.run(server.call_tool("ask_agent", {"question": "Q?"}))

    assert "search_documents" not in tools  # the shops have no documents
    assert tools["ask_agent"].annotations is not None
    assert tools["ask_agent"].annotations.open_world_hint  # it may ask hosted models
    assert find("x") == {
        "studies": [
            {
                "table": "analysis.scenarios",
                "what": "a day with every price moved",
                "holds": ["change_pct"],
            }
        ],
        "models": [
            {"tool": "predict_demand", "what": "Tickets an hour.", "inputs": ["weekday", "hour"]}
        ],
    }
    assert "analysis.scenarios" in str(sources) and "predict_demand" in str(sources)
    told = replied(reply)
    assert told["steps"][0] == {"id": "s1", "tool": "data", "ask": "Top state?", "sql": TOP,
                                "prediction": None, "skipped": None}  # fmt: skip
    assert told["steps"][1]["prediction"] == {"model": "review", "item": {"country": "Kenya"}}
    assert told["answered"] and told["sql"] == TOP and "Chiapas [sql]." in str(answer)


def test_mcp_with_the_agent_offers_it_as_a_tool(
    stood_in: Callable[[Script], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.server.mcpserver import MCPServer

    stood_in(lambda shape: {})
    served: list[list[str]] = []

    @contextmanager
    def session(*args: Any, **kwargs: Any) -> Iterator[tuple[Any, str]]:
        yield type("Stub", (), {"ask": staticmethod(lambda question: None)})(), "stub"

    def run(self: MCPServer, transport: str, **options: Any) -> None:
        served.append([t.name for t in asyncio.run(self.list_tools())])

    monkeypatch.setenv(
        "MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mcp-mlflow.db').as_posix()}"
    )
    monkeypatch.setattr(cli, "agent_session", session)
    monkeypatch.setattr(MCPServer, "run", run)

    result = CliRunner().invoke(cli.app, ["mcp", "--agent"])

    assert result.exit_code == 0, result.output
    assert served[0][-3:] == ["ask_agent", "explore_segment", "map_layer"]


def test_mcp_sql_keeps_its_guardrails_whoever_calls(session: duckdb.DuckDBPyConnection) -> None:
    server = mcp_server(session)

    result = asyncio.run(server.call_tool(
        "query_tables", {"sql": "SELECT state FROM clean.mexico_production ORDER BY 1"}
    ))  # fmt: skip

    assert json.loads(result.content[0].text) == {
        "columns": ["state"],
        "rows": [["Chiapas"], ["Puebla"]],
        "truncated": False,
        "notes": [],
    }
    assert result.structured_content == json.loads(result.content[0].text)
    with pytest.raises(ToolError, match="Only SELECT may run; this is COPY"):
        asyncio.run(server.call_tool("query_tables", {"sql": "COPY (SELECT 1) TO 'x.csv'"}))


def test_mcp_predictions_and_passages(session: duckdb.DuckDBPyConnection) -> None:
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=PREDICTED)

    server = mcp_server(session, respond)
    item = {"item": {"shop": "Almanegra", "bag_grams": 250}}

    priced = json.loads(asyncio.run(server.call_tool("predict_offer", item)).content[0].text)
    found = asyncio.run(server.call_tool("search_documents", {"question": "Why?", "k": 50}))

    assert priced["item"] == {"shop": "almanegra", "bag_grams": 250.0}
    assert priced["prediction"] == 1324.93
    assert sent == [{"shop": "almanegra", "bag_grams": 250.0}]
    passages = [json.loads(c.text) for c in found.content]
    assert len(passages) == 10  # asked for 50: capped
    assert passages[0]["source"] == "FAO, page 12"
    with pytest.raises(ToolError, match="greater than 0"):
        asyncio.run(server.call_tool("predict_offer", {"item": {"shop": "a", "bag_grams": -5}}))


def test_mcp_says_when_the_prediction_service_fails(session: duckdb.DuckDBPyConnection) -> None:
    server = mcp_server(session, lambda request: httpx.Response(503))

    with pytest.raises(ToolError, match="The prediction service failed"):
        asyncio.run(server.call_tool("predict_review", {"item": {"country": "Ethiopia"}}))


BY_STATE = "SELECT state, production_t FROM clean.mexico_production ORDER BY state"


def test_mcp_draws_a_result_as_its_shape_asks(session: duckdb.DuckDBPyConnection) -> None:
    server = mcp_server(session)

    drawn = asyncio.run(server.call_tool("draw", {"sql": BY_STATE}))

    image, summary = drawn.content
    assert image.type == "image" and image.mime_type == "image/png"
    told = json.loads(summary.text)
    assert {k: told[k] for k in ("chart", "rows", "truncated")} == {
        "chart": {"kind": "bar", "x": "state", "y": "production_t"},
        "rows": 2,
        "truncated": False,
    }
    # Two rows: the spec comes too, for a client that draws its own charts.
    assert told["vega_lite"]["mark"] == {"type": "bar"}


def test_mcp_draws_the_chart_the_client_asks_for_if_the_result_can_carry_it(
    session: duckdb.DuckDBPyConnection,
) -> None:
    from mlops_core.explore.charts import Areas

    shapes = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": s[:2], "name": s},
         "geometry": {"type": "Polygon", "coordinates": [[[0, i], [1, i], [1, i + 1], [0, i]]]}}
        for i, s in enumerate(["Chiapas", "Puebla"])
    ]}  # fmt: skip
    server = mcp_server(session, areas=Areas("state_id", "state", shapes))

    mapped = asyncio.run(server.call_tool("draw", {"sql": BY_STATE}))
    listed = asyncio.run(server.call_tool("draw", {"sql": BY_STATE, "chart": {"kind": "table"}}))

    assert json.loads(mapped.content[1].text)["chart"] == {"kind": "areas", "y": "production_t"}
    assert json.loads(listed.content[0].text)["values"] == [["Chiapas", 391690.56], ["Puebla", 1.0]]
    with pytest.raises(ToolError, match="y must be a number; 'state' is not"):
        asyncio.run(server.call_tool(
            "draw", {"sql": BY_STATE, "chart": {"kind": "bar", "x": "production_t", "y": "state"}}
        ))  # fmt: skip
    with pytest.raises(ToolError, match="Only SELECT may run"):
        asyncio.run(server.call_tool("draw", {"sql": "DROP VIEW clean.mexico_production"}))


def test_mcp_cells_travel_as_json() -> None:
    from mlops_core.agent.mcp_server import _plain

    assert _plain(date(2026, 9, 25)) == "2026-09-25"
    assert _plain(Decimal("1.5")) == "1.5"
    assert _plain(3) == 3


def test_the_mcp_command_serves_on_stdio(
    stood_in: Callable[[Script], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.server.mcpserver import MCPServer

    stood_in(lambda shape: {})
    served: list[tuple[str, list[str]]] = []

    def run(self: MCPServer, transport: str, **options: Any) -> None:
        served.append((transport, [t.name for t in asyncio.run(self.list_tools())]))
        assert options in ({}, {"host": "127.0.0.1", "port": 9000})  # never beyond this machine

    monkeypatch.setattr(MCPServer, "run", run)

    result = CliRunner().invoke(cli.app, ["mcp"])
    over_http = CliRunner().invoke(cli.app, ["mcp", "--http", "--port", "9000"])

    assert result.exit_code == 0, result.output
    assert over_http.exit_code == 0, over_http.output
    tools = [
        "query_tables",
        "draw",
        "predict_review",
        "predict_offer",
        "predict_green_price",
        *NEW_PREDICTIONS,
        "search_documents",
        "find_sources",
        "explore_segment",
        "map_layer",
    ]
    assert served == [("stdio", tools), ("streamable-http", tools)]


@pytest.mark.parametrize(
    ("citations", "answered"), [(["prediction"], True), ([], False)], ids=["cited", "uncited"]
)
def test_an_answer_that_cites_its_evidence_has_answered(
    session: duckdb.DuckDBPyConnection, citations: list[str], answered: bool
) -> None:
    """The answer prompt's rule: "answered" is false only for an answer that cites nothing."""
    text = "Its chance is 0.58 [prediction]." if citations else "Nothing says."
    generator = Scripted(
        RouteReply=lambda p: {"route": "prediction"},
        ModelChoice=lambda p: {"model": "offer"},
        Offer=lambda p: {"shop": "almanegra", "bag_grams": 250},
        AnswerReply=lambda p: {"text": text, "citations": citations, "answered": False},
    )

    reply = agent(generator, session).ask("Is it likely?")

    assert reply.answered is answered


def test_a_domain_without_documents_gets_an_agent_without_a_library() -> None:
    """A business brings its tables, not a library: its agent searches no index, and a
    question about documents finds no passage, which the agent says."""
    config = load_adapter("coffee/cafe_de_barrio").config

    passages, titles = cli._documents(config, Settings(), embedder=None)  # type: ignore[arg-type]

    assert config.corpus is None
    assert passages("Why does altitude matter?", 8) == []
    assert titles == {}


def test_a_domain_without_documents_is_judged_on_its_tables_and_models(tmp_path: Path) -> None:
    """Its evaluation reads no retrieval questions: it has no passage to retrieve."""
    shop = load_adapter("coffee/cafe_de_barrio").config

    assert cli._retrieval_questions(shop, tmp_path / "absent.jsonl") == []
