"""The agent: a LangGraph workflow over the three tools, with autonomy only where it pays.

A fixed path - route, plan, the tools the plan needs, gate, answer, verify - with loops
only where evidence says they help, each capped: a query that fails goes back for repair
(twice, inside text to SQL), one that ran and found nothing goes back once with the values
it filtered on checked against the data, and an answer that fails verification is written
again once, told what was wrong. A 3-4B model with free rein picks tools badly (the
benchmark saw it); a workflow asks it only small questions, each with a constrained reply.

The gate stands between the tools and the answer. A query that found nothing, a
prediction the API refused and passages not close enough to the question are not
evidence; when nothing is left, the agent says it found no answer without calling the
model to write one - a model handed nothing writes a figure anyway (it did: 148 MXN/kg
from a query that returned null).

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
from mlops_core.agent.model_cards import ModelFinder
from mlops_core.agent.prompts import (
    ANSWER,
    FIX,
    NEEDS,
    PLAN,
    PLAN_AGAIN,
    AnswerReply,
    NeedsReply,
    PlanReply,
    Route,
)
from mlops_core.agent.routing import route
from mlops_core.agent.text_to_sql import Generator, SqlAnswer, write_sql
from mlops_core.agent.tools import PredictionAnswer, cite, described, predict
from mlops_core.agent.verify import cited_ids, problems
from mlops_core.config import SqlGuard

# Chosen by the benchmark (2026-09-25): the one candidate that cleared both bars.
AGENT_GENERATOR = "qwen3.5:4b"
PASSAGES = 8  # dense search had the answer in the top eight for 77% of 108 questions (73% in five)
MAX_ANSWERS = 2  # the first answer and one rewrite
# A passage less similar to the question than this is not evidence. Measured on
# 2026-09-29: the best passage for the 108 retrieval questions scored 0.428 at the lowest
# (0.60 for the fifth percentile), for 20 off-topic questions 0.28 at the median and 0.56
# at the highest (green tea, chocolate). 0.45 keeps 107 of the 108 and stops 18 of the 20;
# what gets through is near enough to the domain for the answer to say it does not answer.
MIN_PASSAGE_SCORE = 0.45
NO_ANSWER = "I found no answer to this in the tables, the models or the documents."

Passages = Callable[[str, int], list[dict[str, Any]]]


class State(TypedDict, total=False):
    question: str
    route: Route
    plan: dict[str, str | None]  # tool -> the part of the question it answers
    models: list[str] | None  # the models it was shown (`ModelFinder`); None, every one
    sql: SqlAnswer | None
    prediction: PredictionAnswer | None
    passages: list[dict[str, Any]]  # what the search returned
    usable: list[dict[str, Any]]  # the passages close enough to the question to be evidence
    abstained: bool  # the gate found no evidence: no answer is written
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
    passages: list[dict[str, Any]]  # every passage the search returned, cited or not
    problems: list[str]  # what verification still found after the rewrite
    # False when the agent said it found no answer instead of giving one.
    answered: bool = True

    @property
    def verified(self) -> bool:
        return not self.problems


class TracedGenerator:
    """A generator whose every call is a span: the prompt in, the reply out, and - for a
    chain of providers - which one answered (or "cache")."""

    def __init__(self, inner: Generator):
        self._inner = inner

    def ask[R: BaseModel](self, prompt: str, reply: type[R]) -> R:
        with mlflow.start_span(name=reply.__name__, span_type="LLM") as span:
            span.set_inputs({"prompt": prompt})
            answer = self._inner.ask(prompt, reply)
            span.set_outputs(answer.model_dump(mode="json"))
            answered_by = getattr(self._inner, "last", None)
            if answered_by:
                span.set_attribute("answered_by", answered_by)
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
    guards: list[SqlGuard] = field(default_factory=list)  # the domain's rules for its SQL
    # The sections a question needs (`dictionary.SchemaLinker`); unset, the whole schema.
    linker: Callable[[str], str] | None = None
    # More SQL writers, each with its own sampling, whose queries vote on the answer
    # (`text_to_sql.voted`); none, the generator's query is the answer.
    voters: list[Generator] = field(default_factory=list)
    # The served models closest to a question, each with its inputs; unset, every model
    # the domain offers, as `routing` lists them.
    models: ModelFinder | None = None

    def __post_init__(self) -> None:
        self.generator = TracedGenerator(self.generator)
        self.voters = [TracedGenerator(voter) for voter in self.voters]
        graph = StateGraph(State)
        for name, step in (
            ("route", self._route),
            ("plan", self._plan),
            ("data", self._data),
            ("prediction", self._prediction),
            ("knowledge", self._knowledge),
            ("gate", self._gate),
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
            ("knowledge", "gate"),
            ("answer", "verify"),
        ):
            graph.add_edge(before, after)
        graph.add_conditional_edges("gate", self._evidenced, ["answer", END])
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
        question = state["question"]
        with _span("route"):
            cards = self.models.closest(question) if self.models is not None else None
            models = (
                "\n".join(card.listing() for card in cards)
                if cards is not None
                else self.routing["models"]
            )
            chosen = route(self.generator, self.routing | {"models": models}, question)
            needs = self._needs(question, models)
        taken, plan = self._planned(question, chosen, needs, models)
        shown = [card.name for card in cards] if cards is not None else None
        return {"route": taken, "plan": plan, "models": shown}

    def _needs(self, question: str, models: str) -> NeedsReply:
        """A second opinion, as two yes-or-no questions: one route of four is the call the
        small model got wrong most (a bag it describes sent to the tables)."""
        prompt = NEEDS.format(subject=self.routing["subject"], models=models, question=question)
        return self.generator.ask(prompt, NeedsReply)

    def _planned(
        self, question: str, chosen: Route, needs: NeedsReply, models: str
    ) -> tuple[Route, dict[str, str | None]]:
        """The route's tools, with what the second opinion adds: a prediction for an item
        the question describes, the tables for a figure computed from their records. One
        route is replaced: an item described, and no figure asked of the tables, sent to
        the tables - its query finds nothing and the answer is left to explain why.
        Otherwise a tool is only ever added, never taken away."""
        if chosen == "data" and needs.predicts and not needs.figures:
            chosen = "prediction"
        if chosen != "mixed":
            plan: dict[str, str | None] = {chosen: question}
        else:
            with _span("plan"):
                prompt = PLAN.format(
                    subject=self.routing["subject"], models=models, question=question
                )
                reply = self.generator.ask(prompt, PlanReply)
                if reply.repeats():  # handed whole to two tools: asked once more
                    reply = self.generator.ask(prompt + PLAN_AGAIN, PlanReply)
                plan = reply.by_tool()
            if not any(plan.values()):  # a plan with no tool: ask the tables and the documents
                plan = {"data": question, "knowledge": question}
        if needs.predicts and not plan.get("prediction"):
            plan["prediction"] = question
        if needs.figures and not plan.get("data") and not plan.get("prediction"):
            plan["data"] = question
        tools = [tool for tool, part in plan.items() if part]
        return (chosen if len(tools) == 1 else "mixed"), plan

    def _plan(self, state: State) -> State:
        """The plan was made with the route; this step only names it in the trace."""
        with _span("plan"):
            return {"plan": state["plan"]}

    def _data(self, state: State) -> State:
        part = state["plan"].get("data")
        if not part:
            return {"sql": None}
        schema = self.linker(part) if self.linker else self.schema
        with _span("data"):
            answer = write_sql(
                self.generator, self.con, schema, part, guards=self.guards, voters=self.voters
            )
            return {"sql": answer}

    def _prediction(self, state: State) -> State:
        part = state["plan"].get("prediction")
        if not part:
            return {"prediction": None}
        with _span("prediction"):
            answer = predict(
                self.generator,
                self.adapter,
                self.api,
                part,
                state["question"],
                state.get("models"),
            )
            return {"prediction": answer}

    def _knowledge(self, state: State) -> State:
        part = state["plan"].get("knowledge")
        if not part:
            return {"passages": []}
        with _span("knowledge"):
            return {"passages": self.passages(part, PASSAGES)}

    def _gate(self, state: State) -> State:
        """Keep what is evidence; abstain when nothing is."""
        usable = [p for p in state["passages"] if p.get("score", 1.0) >= MIN_PASSAGE_SCORE]
        sql, prediction = state["sql"], state["prediction"]
        found = (
            (sql is not None and sql.result is not None and not sql.empty)
            or (prediction is not None and prediction.response is not None)
            or bool(usable)
        )
        return {"usable": usable, "abstained": not found}

    def _evidenced(self, state: State) -> str:
        return END if state["abstained"] else "answer"

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
            if sql.result is None:
                blocks.append(f"The tables could not answer: {sql.error}")
            elif sql.empty:
                blocks.append(f"The tables have nothing for it: the query {sql.sql} found no rows.")
            else:
                blocks.append(f"[sql] Query result (SQL: {sql.sql}):\n{sql.result.as_text()}")
        prediction = state["prediction"]
        if prediction is not None:
            if prediction.response:
                response = prediction.response
                blocks.append(
                    f"[prediction] The {prediction.model} model's prediction for the item "
                    f"{json.dumps(prediction.request)}: {described(response)}"
                )
            else:
                blocks.append(f"No prediction: {prediction.error}")
        for n, passage in enumerate(state["usable"], start=1):
            blocks.append(f"[c{n}] {self._source(passage)}:\n{passage['text']}")
        return "\n\n".join(blocks) or "No tool returned anything."

    def _source(self, passage: dict[str, Any]) -> str:
        return cite(passage, self.documents)

    def _reply(self, state: State) -> Reply:
        """The best answer, with a line per piece of evidence it cites, in the order given;
        or, when the gate found no evidence, that no answer was found - and why."""
        if state["abstained"]:
            return Reply(
                state["question"],
                state["route"],
                " ".join([NO_ANSWER, *_why(state)]),
                [],
                state["sql"],
                state["prediction"],
                [],
                [],
                answered=False,
            )
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
                passage = state["usable"][int(source[1:]) - 1]
                sources.append(f"[{source}] {self._source(passage)}")
        return Reply(
            state["question"],
            state["route"],
            answer.text,
            sources,
            state["sql"],
            state["prediction"],
            state["usable"],
            state["best_problems"],
            # The answer prompt's own rule: "answered" is false only for an answer that
            # cites nothing. Handed a prediction, the small model cited it, gave it, and
            # still called it unanswered - the evidence "does not confirm this as a fact".
            # An estimate is what a question to a model asks for.
            answered=answer.answered or bool(sources),
        )


def _ids(state: State) -> list[str]:
    """The ids of the evidence the gate kept, in the order the answer saw them."""
    sql, prediction = state["sql"], state["prediction"]
    return (
        (["sql"] if sql is not None and sql.result is not None and not sql.empty else [])
        + (["prediction"] if prediction is not None and prediction.response else [])
        + [f"c{n}" for n in range(1, len(state["usable"]) + 1)]
    )


def _why(state: State) -> list[str]:
    """What each tool that ran came back with, when none of it was evidence."""
    reasons = []
    sql, prediction = state["sql"], state["prediction"]
    if sql is not None:
        reasons.append(
            f"The tables: the query failed ({sql.error})."
            if sql.result is None
            else "The tables: the query ran and found nothing."
        )
    if prediction is not None:
        reasons.append(f"The models: no prediction ({prediction.error}).")
    if state["plan"].get("knowledge"):
        reasons.append("The documents: no passage is close enough to the question.")
    return reasons


@contextmanager
def _span(name: str) -> Iterator[None]:
    with mlflow.start_span(name=name):
        yield
