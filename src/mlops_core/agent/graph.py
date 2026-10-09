"""The agent: a LangGraph workflow over the three tools, with autonomy only where it pays.

A fixed path - route, plan, the tools the plan needs, gate, answer, verify - with loops
only where evidence says they help, each capped: a query that fails goes back for repair
(twice, inside text to SQL), one that ran and found nothing goes back once with the values
it filtered on checked against the data, and an answer that fails verification is written
again once, told what was wrong. A 3-4B model with free rein picks tools badly (the
benchmark saw it); a workflow asks it only small questions, each with a constrained reply.

With a planner (`agent.planner`), the first step plans the whole question instead of
routing it: whether it is about the domain at all - declined before any tool runs when it
is not - and the steps that answer it, each with its tool and the steps it needs. The
steps run in rounds: those that need nothing from each other at once, each later one
handed what the steps it needs found (`planner`). A plan that cannot run is asked for once
more, then left for the router.

The gate stands between the tools and the answer. A query that found nothing, a
prediction the API refused and passages not close enough to the question are not
evidence; when nothing is left, the agent says it found no answer without calling the
model to write one - a model handed nothing writes a figure anyway (it did: 148 MXN/kg
from a query that returned null).

Every run is one MLflow trace: a span per step and per call to the model, and the
registry versions of the prompts it sent.
"""

import contextvars
import json
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypedDict

import duckdb
import httpx
import mlflow
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from mlops_core.adapter import DomainAdapter
from mlops_core.agent.dictionary import reads_any, table_sections
from mlops_core.agent.model_cards import ModelCard, ModelFinder
from mlops_core.agent.planner import (
    StepResult,
    from_route_plan,
    known,
    normalised,
    plan_problems,
    route_of,
    sources,
    waves,
)
from mlops_core.agent.prompts import (
    ANSWER,
    FIX,
    HOLDS,
    NEEDS,
    PLAN,
    PLAN_AGAIN,
    PLANNER,
    PLANNER_AGAIN,
    AnswerReply,
    HoldsReply,
    NeedsReply,
    PlannedStep,
    PlanReply,
    ReplyRoute,
    Route,
    StepPlan,
)
from mlops_core.agent.routing import route
from mlops_core.agent.study_cards import NO_STUDIES, StudyFinder
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
OFF_TARGET = "the query's result is not what the question asks: it holds "
OUT_OF_SCOPE = (
    "That is not something I can answer: I answer questions about {subject}, from its "
    "tables, its models and its documents."
)

Passages = Callable[[str, int], list[dict[str, Any]]]


class State(TypedDict, total=False):
    question: str
    route: ReplyRoute
    plan: dict[str, str | None]  # tool -> the part of the question it answers
    models: list[str] | None  # the models it was shown (`ModelFinder`); None, every one
    planned: list[PlannedStep]  # the planner's steps, in the order planned
    steps: list[StepResult]  # what each planned step found, in the same order
    out_of_scope: bool  # the planner found the question about none of the domain
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
    route: ReplyRoute
    text: str
    sources: list[str]  # the cited passages, as a reader can find them
    sql: SqlAnswer | None  # the first query run, when the plan had more than one
    prediction: PredictionAnswer | None  # the first prediction, likewise
    passages: list[dict[str, Any]]  # every passage the search returned, cited or not
    problems: list[str]  # what verification still found after the rewrite
    # False when the agent said it found no answer instead of giving one.
    answered: bool = True
    steps: tuple[StepResult, ...] = ()  # every planned step and what it found (a planner's)

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


@dataclass(frozen=True)
class _Piece:
    """One thing a tool came back with, as the answer reads it: an id when it is evidence."""

    id: str | None
    block: str
    source: str | None = None  # how a reply lists it when the answer cites it


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
    # The studies closest to a question, shown to the steps that choose a tool; unset,
    # none (`agent.study_cards`).
    studies: StudyFinder | None = None
    # Whether a query's result is checked against the question before it is evidence
    # (`agent.check_result`).
    check_result: bool = False
    # Whether the domain has documents to search; a business that brings only its tables
    # has none, and a question routed to them is asked of the tables.
    library: bool = True
    # Whether the question is planned as steps before any tool runs (`agent.planner`),
    # instead of routed.
    planner: bool = False

    def __post_init__(self) -> None:
        self.generator = TracedGenerator(self.generator)
        self.voters = [TracedGenerator(voter) for voter in self.voters]
        graph = StateGraph(State)
        if self.planner:
            graph.add_node("plan", self._plan_steps)
            graph.add_node("execute", self._execute)
            graph.add_edge(START, "plan")
            graph.add_conditional_edges("plan", self._in_scope, ["execute", END])
            edges = [("execute", "gate")]
        else:
            for name, step in (
                ("route", self._route),
                ("plan", self._plan),
                ("data", self._data),
                ("prediction", self._prediction),
                ("knowledge", self._knowledge),
            ):
                graph.add_node(name, step)
            graph.add_edge(START, "route")
            edges = [
                ("route", "plan"),
                ("plan", "data"),
                ("data", "prediction"),
                ("prediction", "knowledge"),
                ("knowledge", "gate"),
            ]
        for name, step in (
            ("gate", self._gate),
            ("answer", self._answer),
            ("verify", self._verify),
        ):
            graph.add_node(name, step)
        for before, after in (*edges, ("answer", "verify")):
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

    def _closest_models(self, question: str) -> tuple[list[ModelCard] | None, str]:
        """The model cards a question is shown, and their listing."""
        cards = self.models.closest(question) if self.models is not None else None
        listing = (
            "\n".join(card.listing() for card in cards)
            if cards is not None
            else self.routing["models"]
        )
        return cards, listing

    def _route(self, state: State) -> State:
        question = state["question"]
        with _span("route"):
            cards, models = self._closest_models(question)
            shown = self.studies.blocks(question) if self.studies is not None else NO_STUDIES
            context = self.routing | {"models": models, "studies": shown["studies"]}
            chosen = route(self.generator, context, question)
            needs = self._needs(question, models, shown["needs_studies"])
        taken, plan = self._planned(question, chosen, needs, models, shown["plan_studies"])
        names = [card.name for card in cards] if cards is not None else None
        return {"route": taken, "plan": plan, "models": names}

    def _needs(self, question: str, models: str, studies: str = "") -> NeedsReply:
        """A second opinion, as two yes-or-no questions: one route of four is the call the
        small model got wrong most (a bag it describes sent to the tables)."""
        prompt = NEEDS.format(
            subject=self.routing["subject"],
            models=models,
            question=question,
            needs_studies=studies,
        )
        return self.generator.ask(prompt, NeedsReply)

    def _planned(
        self, question: str, chosen: Route, needs: NeedsReply, models: str, studies: str = ""
    ) -> tuple[Route, dict[str, str | None]]:
        """The route's tools, with what the second opinion adds: a prediction for an item
        the question describes, the tables for a figure computed from their records. One
        route is replaced: an item described, and no figure asked of the tables, sent to
        the tables - its query finds nothing and the answer is left to explain why.
        Otherwise a tool is only ever added, never taken away - but the documents, in a
        domain with none: what was meant for them goes to the tables."""
        # Not when the router was shown the studies: told that a what-if a study holds is
        # data, it said so, and the second opinion - which a small model reads less well -
        # turned five such answers into predictions (2026-10-07). It still adds one.
        if chosen == "data" and needs.predicts and not needs.figures and self.studies is None:
            chosen = "prediction"
        if chosen == "knowledge" and not self.library:
            # Seen 2026-10-07: a business's agent, which has no library, sent a question
            # its tables answer to the documents, and found nothing.
            chosen = "data"
        if chosen != "mixed":
            plan: dict[str, str | None] = {chosen: question}
        else:
            with _span("plan"):
                prompt = PLAN.format(
                    subject=self.routing["subject"],
                    models=models,
                    question=question,
                    plan_studies=studies,
                )
                reply = self.generator.ask(prompt, PlanReply)
                if reply.repeats():  # handed whole to two tools: asked once more
                    reply = self.generator.ask(prompt + PLAN_AGAIN, PlanReply)
                plan = reply.by_tool()
            if not any(plan.values()):  # a plan with no tool: ask the tables and the documents
                plan = {"data": question, "knowledge": question}
            if not self.library and plan.get("knowledge"):
                plan["data"] = plan.get("data") or plan["knowledge"]
                plan["knowledge"] = None
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

    def _plan_steps(self, state: State) -> State:
        """The planner's plan: whether the question is about the domain, and its steps. A
        plan that cannot run is asked for again, told why; one that still cannot is left,
        and the router routes the question as it would without a planner."""
        question = state["question"]
        with _span("plan"):
            cards, models = self._closest_models(question)
            shown = self.studies.blocks(question) if self.studies is not None else NO_STUDIES
            prompt = PLANNER.format(
                subject=self.routing["subject"],
                sources=sources(models, self.routing["topics"], shown["studies"], self.library),
                tables=self.routing["tables"],
                question=question,
            )
            plan = normalised(self.generator.ask(prompt, StepPlan), self.library)
            wrong = plan_problems(plan)
            if wrong:
                again = PLANNER_AGAIN.format(problems="\n".join(f"- {w}" for w in wrong))
                plan = normalised(self.generator.ask(prompt + again, StepPlan), self.library)
                wrong = plan_problems(plan)
        names = [card.name for card in cards] if cards is not None else None
        if wrong:
            routed = self._route(state)
            return routed | {"planned": from_route_plan(routed["plan"]), "out_of_scope": False}
        if not plan.in_scope:
            return {"route": "none", "plan": {}, "planned": [], "models": names,
                    "out_of_scope": True}  # fmt: skip
        return {
            "route": route_of(plan.steps),  # type: ignore[typeddict-item]
            "plan": _by_tool(plan.steps),
            "planned": plan.steps,
            "models": names,
            "out_of_scope": False,
        }

    def _in_scope(self, state: State) -> str:
        return END if state["out_of_scope"] else "execute"

    def _execute(self, state: State) -> State:
        """Every planned step, in rounds: a round's steps at once, each handed what the
        steps it needs found. A step whose needs found nothing is not run: it would be
        asked with nothing to go on."""
        planned = state["planned"]
        predictions = sum(step.tool == "prediction" for step in planned)
        results: dict[str, StepResult] = {}
        for wave in waves(planned):
            ready = []
            for step in wave:
                empty = [used for used in step.uses if not results[used].found]
                if empty:
                    why = f"it needs what {', '.join(empty)} found, and that found nothing"
                    results[step.id] = StepResult(step, skipped=why)
                else:
                    ready.append(step)
            results |= self._run_round(ready, results, state["question"], state, predictions)
        ordered = [results[step.id] for step in planned]
        passages: dict[str, dict[str, Any]] = {}
        for result in ordered:
            for passage in result.passages:
                passages.setdefault(str(passage.get("chunk_id", id(passage))), passage)
        return {
            "steps": ordered,
            "sql": next((r.sql for r in ordered if r.sql is not None), None),
            "prediction": next((r.prediction for r in ordered if r.prediction is not None), None),
            "passages": list(passages.values()),
        }

    def _run_round(
        self,
        steps: Sequence[PlannedStep],
        results: dict[str, StepResult],
        question: str,
        state: State,
        predictions: int,
    ) -> dict[str, StepResult]:
        """One round's steps: at once when there are several. Each runs in a copy of the
        trace's context, so its spans are the agent's; a local model still takes its calls
        one at a time (`rag.llm.LOCAL_TURN`)."""
        found = dict(results)
        if len(steps) <= 1:
            return {s.id: self._run_step(s, found, question, state, predictions) for s in steps}
        with ThreadPoolExecutor(max_workers=len(steps)) as pool:
            futures = {
                step.id: pool.submit(
                    contextvars.copy_context().run,
                    self._run_step,
                    step,
                    found,
                    question,
                    state,
                    predictions,
                )
                for step in steps
            }
            return {name: future.result() for name, future in futures.items()}

    def _run_step(
        self,
        step: PlannedStep,
        results: dict[str, StepResult],
        question: str,
        state: State,
        predictions: int,
    ) -> StepResult:
        handed = known(results, step.uses)
        with _span(f"step {step.id}: {step.tool}"):
            if step.tool == "data":
                part = f"{step.ask}\n\n{handed}" if handed else step.ask
                return StepResult(step, sql=self._query(part))
            if step.tool == "prediction":
                # The item is read from the asker's own words when the plan has one
                # prediction - a plan restates its part, and a detail it drops is a field
                # left empty - and from the step's when it has more, each about its own
                # item. What the steps it needs found is part of what was said.
                words = question if predictions == 1 else step.ask
                asked = f"{words}\n\n{handed}" if handed else words
                answer = predict(
                    self.generator,
                    self.adapter,
                    self.api,
                    step.ask,
                    asked,
                    state.get("models"),
                )
                return StepResult(step, prediction=answer)
            return StepResult(step, passages=self.passages(step.ask, PASSAGES))

    def _data(self, state: State) -> State:
        part = state["plan"].get("data")
        if not part:
            return {"sql": None}
        with _span("data"):
            return {"sql": self._query(part)}

    def _query(self, part: str) -> SqlAnswer:
        """A query written for `part` - checked against it, when the domain asks, before
        its result is evidence."""
        schema = self.linker(part) if self.linker else self.schema
        answer = write_sql(
            self.generator, self.con, schema, part, guards=self.guards, voters=self.voters
        )
        if self.check_result and answer.result is not None and not answer.empty:
            instead = self._holds_instead(part, answer)
            if instead is not None:
                answer = SqlAnswer(answer.sql, None, OFF_TARGET + instead, answer.attempts)
        return answer

    def _holds_instead(self, question: str, answer: SqlAnswer) -> str | None:
        """What a query's result holds in place of what `question` asks; None if it holds
        what was asked."""
        assert answer.result is not None
        read = [
            section
            for view, section in table_sections(self.schema).items()
            if reads_any(answer.sql, [view])
        ]
        prompt = HOLDS.format(
            subject=self.routing["subject"],
            tables="\n\n".join(read),
            question=question,
            sql=answer.sql,
            result=answer.result.as_text(),
        )
        with _span("holds"):
            reply = self.generator.ask(prompt, HoldsReply)
        return None if reply.holds else (reply.instead.strip() or "something else")

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
        found = any(piece.id for piece in _pieces(state)) or bool(usable)
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
        blocks = [piece.block for piece in _pieces(state)]
        for n, passage in enumerate(state["usable"], start=1):
            blocks.append(f"[c{n}] {self._source(passage)}:\n{passage['text']}")
        return "\n\n".join(blocks) or "No tool returned anything."

    def _source(self, passage: dict[str, Any]) -> str:
        return cite(passage, self.documents)

    def _reply(self, state: State) -> Reply:
        """The best answer, with a line per piece of evidence it cites, in the order given;
        or, when the gate found no evidence, that no answer was found - and why; or, for a
        question about none of the domain, that it is not one to answer here."""
        steps = tuple(state.get("steps", ()))
        if state.get("out_of_scope"):
            text = OUT_OF_SCOPE.format(subject=self.routing["subject"])
            return Reply(state["question"], "none", text, [], None, None, [], [], False)
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
                steps=steps,
            )
        answer = state["best"]
        cited = cited_ids(answer.text, answer.citations)
        lines = {piece.id: piece.source for piece in _pieces(state) if piece.id}
        sources = []
        for source in _ids(state):
            if source not in cited:
                continue
            if source in lines:
                sources.append(str(lines[source]))
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
            steps=steps,
        )


def _by_tool(steps: Sequence[PlannedStep]) -> dict[str, str | None]:
    """A plan's steps as the router's plan is kept: each tool with what it is asked."""
    plan: dict[str, str | None] = {}
    for step in steps:
        asked = plan.get(step.tool)
        plan[step.tool] = f"{asked} {step.ask}" if asked else step.ask
    return plan


def _pieces(state: State) -> list[_Piece]:
    """What the tools came back with, in the order the answer reads it: the planner's
    steps, each by its own id (`sql`, `sql2`, `prediction`...) and, when there are several,
    labelled with what it was asked; or the router's one query and one prediction."""
    if "steps" not in state:
        pieces = []
        if state.get("sql") is not None:
            pieces.append(_sql_piece("sql", state["sql"], ""))  # type: ignore[arg-type]
        if state.get("prediction") is not None:
            pieces.append(_prediction_piece("prediction", state["prediction"], ""))  # type: ignore[arg-type]
        return pieces
    steps = state["steps"]
    counts = {"data": 0, "prediction": 0}
    pieces = []
    for result in steps:
        label = f' for "{result.step.ask}"' if len(steps) > 1 else ""
        if result.skipped is not None:
            pieces.append(_Piece(None, f"Not looked up{label}: {result.skipped}."))
        elif result.sql is not None:
            counts["data"] += 1
            name = "sql" if counts["data"] == 1 else f"sql{counts['data']}"
            pieces.append(_sql_piece(name, result.sql, label))
        elif result.prediction is not None:
            counts["prediction"] += 1
            n = counts["prediction"]
            name = "prediction" if n == 1 else f"prediction{n}"
            pieces.append(_prediction_piece(name, result.prediction, label))
    return pieces


def _sql_piece(name: str, sql: SqlAnswer, label: str) -> _Piece:
    if sql.result is None:
        return _Piece(None, f"The tables could not answer{label}: {sql.error}")
    if sql.empty:
        return _Piece(None, f"The tables have nothing{label or ' for it'}: the query {sql.sql} "
                            "found no rows.")  # fmt: skip
    block = f"[{name}] Query result{label} (SQL: {sql.sql}):\n{sql.result.as_text()}"
    return _Piece(name, block, f"[{name}] the tables, with the query below")


def _prediction_piece(name: str, prediction: PredictionAnswer, label: str) -> _Piece:
    if not prediction.response:
        return _Piece(None, f"No prediction{label}: {prediction.error}")
    block = (
        f"[{name}] The {prediction.model} model's prediction{label} for the item "
        f"{json.dumps(prediction.request)}: {described(prediction.response)}"
    )
    version = prediction.response.get("model_version", "?")
    return _Piece(name, block, f"[{name}] the {prediction.model} model, v{version}")


def _ids(state: State) -> list[str]:
    """The ids of the evidence the gate kept, in the order the answer saw them."""
    return [piece.id for piece in _pieces(state) if piece.id] + [
        f"c{n}" for n in range(1, len(state["usable"]) + 1)
    ]


def _why(state: State) -> list[str]:
    """What each tool that ran came back with, when none of it was evidence."""
    if "steps" in state:
        return [_why_step(result) for result in state["steps"]]
    reasons = []
    sql, prediction = state["sql"], state["prediction"]
    if sql is not None:
        reasons.append(_why_sql(sql))
    if prediction is not None:
        reasons.append(f"The models: no prediction ({prediction.error}).")
    if state["plan"].get("knowledge"):
        reasons.append("The documents: no passage is close enough to the question.")
    return reasons


def _why_step(result: StepResult) -> str:
    if result.skipped is not None:
        return f"{result.step.id}: not looked up ({result.skipped})."
    if result.sql is not None:
        return f"{result.step.id}: {_why_sql(result.sql)}"
    if result.prediction is not None:
        return f"{result.step.id}: the models: no prediction ({result.prediction.error})."
    return f"{result.step.id}: the documents: no passage is close enough to the question."


def _why_sql(sql: SqlAnswer) -> str:
    if sql.error is not None and sql.error.startswith(OFF_TARGET):
        return f"The tables: {sql.error}."
    if sql.result is None:
        return f"The tables: the query failed ({sql.error})."
    return "The tables: the query ran and found nothing."


@contextmanager
def _span(name: str) -> Iterator[None]:
    with mlflow.start_span(name=name):
        yield
