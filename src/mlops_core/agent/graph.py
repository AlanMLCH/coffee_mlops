"""The agent: a LangGraph workflow over the three tools, with autonomy only where it pays.

A fixed path - route, plan, the tools the plan needs, answer, verify - with loops only
where evidence says they help, each capped: a query that fails goes back for repair
(twice, inside text to SQL), and an answer that fails verification is written again
once, told what was wrong. A 3-4B model with free rein picks tools badly (the benchmark
saw it); a workflow asks it only small questions, each with a constrained reply.

Every run is one MLflow trace: a span per step and per call to the model, and the
registry versions of the prompts it sent.
"""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypedDict

import duckdb
import httpx
import mlflow
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from mlops_core.adapter import DomainAdapter
from mlops_core.agent.prompts import ANSWER, FIX, PLAN, AnswerReply, PlanReply, Route
from mlops_core.agent.routing import route
from mlops_core.agent.text_to_sql import Generator, SqlAnswer, write_sql
from mlops_core.agent.tools import PredictionAnswer, predict
from mlops_core.agent.verify import cited_ids, problems

# Chosen by the benchmark (2026-09-25): the one candidate that cleared both bars.
AGENT_GENERATOR = "qwen3.5:4b"
PASSAGES = 5  # dense search had the answer in the top five for three questions in four
MAX_ANSWERS = 2  # the first answer and one rewrite

Passages = Callable[[str, int], list[dict[str, Any]]]


class State(TypedDict, total=False):
    question: str
    route: Route
    plan: dict[str, str | None]  # tool -> the part of the question it answers
    sql: SqlAnswer | None
    prediction: PredictionAnswer | None
    passages: list[dict[str, Any]]
    evidence: str
    answer: AnswerReply
    problems: list[str]
    answers: int
    # The answer with the fewest problems so far: a rewrite that breaks more than it
    # fixes is not kept.
    best: AnswerReply
    best_problems: list[str]


@dataclass(frozen=True)
class Reply:
    """What the agent says, and everything needed to check it."""

    question: str
    route: Route
    text: str
    sources: list[str]  # the cited passages, as a reader can find them
    sql: SqlAnswer | None
    prediction: PredictionAnswer | None
    problems: list[str]  # what verification still found after the rewrite

    @property
    def verified(self) -> bool:
        return not self.problems


class TracedGenerator:
    """A generator whose every call is a span: the prompt in, the reply out."""

    def __init__(self, inner: Generator):
        self._inner = inner

    def ask[R: BaseModel](self, prompt: str, reply: type[R]) -> R:
        with mlflow.start_span(name=reply.__name__, span_type="LLM") as span:
            span.set_inputs({"prompt": prompt})
            answer = self._inner.ask(prompt, reply)
            span.set_outputs(answer.model_dump(mode="json"))
            return answer


@dataclass
class Agent:
    generator: Generator
    adapter: DomainAdapter
    con: duckdb.DuckDBPyConnection
    schema: str  # the data dictionary's sections, for text to SQL
    routing: dict[str, str]  # what the router is told about each tool
    passages: Passages
    api: httpx.Client
    documents: dict[str, dict[str, Any]]  # document id -> title, publisher, year
    prompt_uris: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.generator = TracedGenerator(self.generator)
        graph = StateGraph(State)
        for name, step in (
            ("route", self._route),
            ("plan", self._plan),
            ("data", self._data),
            ("prediction", self._prediction),
            ("knowledge", self._knowledge),
            ("answer", self._answer),
            ("verify", self._verify),
        ):
            graph.add_node(name, step)
        graph.add_edge(START, "route")
        for before, after in (
            ("route", "plan"),
            ("plan", "data"),
            ("data", "prediction"),
            ("prediction", "knowledge"),
            ("knowledge", "answer"),
            ("answer", "verify"),
        ):
            graph.add_edge(before, after)
        graph.add_conditional_edges("verify", self._again, ["answer", END])
        self._graph = graph.compile()

    def ask(self, question: str) -> Reply:
        """Answer one question, as one trace."""
        with mlflow.start_span(name="agent", span_type="AGENT") as root:
            root.set_inputs({"question": question})
            mlflow.update_current_trace(
                tags={f"prompt.{name}": uri for name, uri in self.prompt_uris.items()}
            )
            state: State = self._graph.invoke({"question": question, "answers": 0})  # type: ignore[assignment]
            reply = self._reply(state)
            root.set_outputs({"text": reply.text, "problems": reply.problems})
            return reply

    # --- The steps -----------------------------------------------------------------------

    def _route(self, state: State) -> State:
        with _span("route"):
            return {"route": route(self.generator, self.routing, state["question"])}

    def _plan(self, state: State) -> State:
        """Which part of the question each tool answers. One tool: all of it."""
        question, chosen = state["question"], state["route"]
        if chosen != "mixed":
            return {"plan": {chosen: question}}
        with _span("plan"):
            prompt = PLAN.format(subject=self.routing["subject"], question=question)
            parts = self.generator.ask(prompt, PlanReply).model_dump()
        if not any(parts.values()):  # a plan that uses no tool: ask the tables and the documents
            parts = {"data": question, "knowledge": question}
        return {"plan": parts}

    def _data(self, state: State) -> State:
        part = state["plan"].get("data")
        if not part:
            return {"sql": None}
        with _span("data"):
            return {"sql": write_sql(self.generator, self.con, self.schema, part)}

    def _prediction(self, state: State) -> State:
        part = state["plan"].get("prediction")
        if not part:
            return {"prediction": None}
        with _span("prediction"):
            return {"prediction": predict(self.generator, self.adapter, self.api, part)}

    def _knowledge(self, state: State) -> State:
        part = state["plan"].get("knowledge")
        if not part:
            return {"passages": []}
        with _span("knowledge"):
            return {"passages": self.passages(part, PASSAGES)}

    def _answer(self, state: State) -> State:
        evidence = self._evidence(state)
        prompt = ANSWER.format(
            subject=self.routing["subject"], question=state["question"], evidence=evidence
        )
        if state.get("problems"):
            prompt += FIX.format(text=state["answer"].text, problems="\n".join(state["problems"]))
        with _span("answer"):
            answer = self.generator.ask(prompt, AnswerReply)
        return {"answer": answer, "evidence": evidence, "answers": state["answers"] + 1}

    def _verify(self, state: State) -> State:
        answer = state["answer"]
        evidence = state["question"] + "\n" + state["evidence"]
        found = problems(answer.text, answer.citations, evidence, _ids(state))
        if "best" in state and len(state["best_problems"]) <= len(found):
            return {"problems": found}
        return {"problems": found, "best": answer, "best_problems": found}

    def _again(self, state: State) -> str:
        return "answer" if state["problems"] and state["answers"] < MAX_ANSWERS else END

    # --- Evidence and the reply ----------------------------------------------------------

    def _evidence(self, state: State) -> str:
        """What the answer may use, each tool's output labelled with where it came from."""
        blocks = []
        sql = state["sql"]
        if sql is not None:
            blocks.append(
                f"[sql] Query result (SQL: {sql.sql}):\n{sql.result.as_text()}"
                if sql.result
                else f"The tables could not answer: {sql.error}"
            )
        prediction = state["prediction"]
        if prediction is not None:
            if prediction.response:
                response = prediction.response
                blocks.append(
                    f"[prediction] The {prediction.model} model's prediction for the item "
                    f"{json.dumps(prediction.request)}: {response['target']} = "
                    f"{response['prediction']:.2f}"
                )
            else:
                blocks.append(f"No prediction: {prediction.error}")
        for n, passage in enumerate(state["passages"], start=1):
            blocks.append(f"[c{n}] {self._source(passage)}:\n{passage['text']}")
        return "\n\n".join(blocks) or "No tool returned anything."

    def _source(self, passage: dict[str, Any]) -> str:
        document = self.documents.get(passage["document_id"], {})
        title = document.get("title", passage["document_id"])
        year = f" ({document['year']})" if document.get("year") else ""
        where = (
            f"section '{passage['part_title']}'"
            if passage.get("part_title")
            else f"page {passage['part']}"
        )
        return f'{document.get("publisher", "")}, "{title}"{year}, {where}'.lstrip(", ")

    def _reply(self, state: State) -> Reply:
        """The best answer, with a line per piece of evidence it cites, in the order given."""
        answer = state["best"]
        cited = cited_ids(answer.text, answer.citations)
        sources = []
        for source in _ids(state):
            if source not in cited:
                continue
            if source == "sql":
                sources.append("[sql] the tables, with the query below")
            elif source == "prediction" and state["prediction"] is not None:
                version = (state["prediction"].response or {}).get("model_version", "?")
                sources.append(f"[prediction] the {state['prediction'].model} model, v{version}")
            else:
                passage = state["passages"][int(source[1:]) - 1]
                sources.append(f"[{source}] {self._source(passage)}")
        return Reply(
            state["question"],
            state["route"],
            answer.text,
            sources,
            state["sql"],
            state["prediction"],
            state["best_problems"],
        )


def _ids(state: State) -> list[str]:
    """The ids of the evidence the tools returned, in the order the answer saw them."""
    sql, prediction = state["sql"], state["prediction"]
    return (
        (["sql"] if sql is not None and sql.result is not None else [])
        + (["prediction"] if prediction is not None and prediction.response else [])
        + [f"c{n}" for n in range(1, len(state["passages"]) + 1)]
    )


@contextmanager
def _span(name: str) -> Iterator[None]:
    with mlflow.start_span(name=name):
        yield
