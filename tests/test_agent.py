"""The agent: its steps, its checks, and the record it leaves.

A scripted generator stands in for the model, answering by the shape it is asked for;
the prediction API is a mock transport; retrieval is a function returning passages.
What is under test is the workflow - which step runs, what each is shown, when an
answer is written again and which one is kept - not a model.
"""

import asyncio
import json
from collections.abc import Callable, Iterator
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
from mlops_core.adapter import domain_dir
from mlops_core.agent import registry
from mlops_core.agent.graph import Agent
from mlops_core.agent.prompts import PROMPTS
from mlops_core.agent.registry import PREFIX, register_prompts
from mlops_core.agent.sql import read_only
from mlops_core.agent.tools import predict
from mlops_core.agent.verify import cited_ids, figures, problems
from mlops_core.config import CHUNKS_TABLE, DOCUMENTS_TABLE
from mlops_core.data.corpus import CHUNKS_COLUMNS
from mlops_core.rag import llm
from mlops_core.rag.llm import ollama_client
from mlops_core.rag.vectors import build_index, embedding_table
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


class Scripted:
    """Answers each call by the name of the shape it is asked for."""

    def __init__(self, **answers: Callable[[str], dict[str, Any]] | list[dict[str, Any]]):
        self.answers = {k: iter(v) if isinstance(v, list) else v for k, v in answers.items()}
        self.prompts: list[tuple[str, str]] = []

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply:
        self.prompts.append((reply.__name__, prompt))
        answer = self.answers[reply.__name__]
        return reply.model_validate(
            next(answer) if isinstance(answer, Iterator) else answer(prompt)
        )

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
) -> Agent:
    return Agent(
        generator,
        domains.coffee.adapter(),
        session,
        "## `clean.mexico_production` — coffee grown",
        {"subject": "coffee", "tables": "", "models": "", "topics": ""},
        lambda question, k: [PASSAGE],
        api(responder or (lambda request: httpx.Response(200, json=PREDICTED))),
        {"fao": {"title": "Arabica coffee manual", "publisher": "FAO", "year": 2005}},
        {"answer": "prompts:/mlops-agent-answer/1"},
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
    assert "- offer (predicts price_mxn_per_kg): The price" in generator.asked("ModelChoice")[0]
    assert "A model predicts The price per kilogram" in generator.asked("Offer")[0]


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
        PlanReply=lambda p: {"data": "Top state?", "prediction": None, "knowledge": "Why?"},
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
        PlanReply=lambda p: {"data": None, "prediction": None, "knowledge": None},
        SqlReply=lambda p: {"sql": "SELECT 1 AS one"},
        AnswerReply=lambda p: {"text": "Yes [c1].", "citations": ["c1"]},
    )

    reply = agent(generator, session).ask("Something?")

    assert reply.sql is not None
    assert "[c1]" in generator.asked("AnswerReply")[0]


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
                {"chunks_partition": partition.name, "embedding_model": "e@1"}, "s")  # fmt: skip
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
            if request.url.path == "/api/embed":
                return httpx.Response(200, json={"embeddings": [[0.5, 0.5, 0.5, 0.5]]})
            reply = script(json.loads(request.content)["format"]["properties"])
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
        if "route" in shape:
            return {"route": "knowledge"}
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
        if "route" in shape:
            return {"route": "mixed"}
        if "knowledge" in shape:
            return {"data": "How many chunks?", "prediction": "Price?", "knowledge": None}
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


def test_a_prediction_that_failed_is_evidence_too(session: duckdb.DuckDBPyConnection) -> None:
    """The answer is told the service failed, instead of being told nothing."""
    generator = Scripted(
        RouteReply=lambda p: {"route": "prediction"},
        ModelChoice=lambda p: {"model": "review"},
        Lot=lambda p: {"country": "Ethiopia"},
        AnswerReply=lambda p: {"text": "No score could be predicted.", "citations": []},
    )

    reply = agent(generator, session, lambda r: httpx.Response(503)).ask("Score?")

    assert "No prediction: The prediction service failed" in generator.asked("AnswerReply")[0]
    assert reply.verified and reply.sources == []


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
        ],
        "sql_questions.jsonl": [{"id": "production-01", "question": "Top state?", "sql": TOP}],
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
            if "route" in shape:
                return {"route": "mixed"}
            if "knowledge" in shape:
                return {"data": "Top state?", "prediction": None, "knowledge": "Why altitude?"}
            if "sql" in shape:
                return {"sql": sql}
            return {"text": "It is so [sql] [c1].", "citations": ["sql", "c1"]}

        stood_in(script)
        return CliRunner().invoke(cli.app, ["agent", "evaluate"])

    first = run(TOP)
    second = run(TOP.replace("DESC", "ASC"))

    assert first.exit_code == 0, first.output
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
) -> Any:
    from mlops_core.agent.mcp_server import build_server

    return build_server(
        domains.coffee.adapter(),
        session,
        "## `clean.mexico_production` — coffee grown",
        lambda question, k: [PASSAGE] * k,
        api(respond or (lambda request: httpx.Response(200, json=PREDICTED))),
        lambda passage: f"FAO, page {passage['part']}",
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
        "search_documents",
    }
    assert all(t.annotations is not None and t.annotations.read_only_hint for t in tools.values())
    # A prediction tool's input is the model's own request body, descriptions included.
    offer = json.dumps(tools["predict_offer"].input_schema)
    assert "The roaster, as the catalogues name it" in offer
    assert tools["predict_offer"].description.startswith("Predict the price per kilogram")
    assert "`clean.mexico_production`" in next(iter(resource)).content


def test_mcp_sql_keeps_its_guardrails_whoever_calls(session: duckdb.DuckDBPyConnection) -> None:
    server = mcp_server(session)

    result = asyncio.run(server.call_tool(
        "query_tables", {"sql": "SELECT state FROM clean.mexico_production ORDER BY 1"}
    ))  # fmt: skip

    assert json.loads(result.content[0].text) == {
        "columns": ["state"],
        "rows": [["Chiapas"], ["Puebla"]],
        "truncated": False,
    }
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

    def run(self: MCPServer, transport: str) -> None:
        served.append((transport, [t.name for t in asyncio.run(self.list_tools())]))

    monkeypatch.setattr(MCPServer, "run", run)

    result = CliRunner().invoke(cli.app, ["mcp"])

    assert result.exit_code == 0, result.output
    assert served == [
        (
            "stdio",
            [
                "query_tables",
                "predict_review",
                "predict_offer",
                "predict_green_price",
                "search_documents",
            ],
        )
    ]
