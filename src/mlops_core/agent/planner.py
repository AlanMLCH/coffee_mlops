"""A question planned before any tool runs: whether it is about the domain, the steps that
answer it, and what each step needs from another (`agent.planner`).

The router chose a route, and only a `mixed` route was split into parts: a question the
router called `data` was never split, and its prediction never made. And the parts did
not know each other - "how long would an order take then, with that many tickets" was
asked of the model without the number the tables had just found. Here one call plans the
whole question as steps, each with the one source that finds it and the earlier steps it
needs (ReWOO, Xu et al. 2023; LLMCompiler, Kim et al. 2023): the steps that need nothing
from each other run at once, a step that needs another runs after it and is handed what
it found, and the answer joins them. A question about none of the domain is declined
before any tool runs.

The plan is checked before it runs - ids that repeat, a step that needs one not planned
before it, too many steps - and asked for once more with what was wrong; a plan that is
still wrong is not run, and the router answers as it would without a planner.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from mlops_core.agent.prompts import PlannedStep, StepPlan, Tool
from mlops_core.agent.text_to_sql import SqlAnswer
from mlops_core.agent.tools import PredictionAnswer, described

MAX_STEPS = 5  # past this a plan is restating the question, not splitting it
# What a step is handed of what another found: a result's first rows, a passage's start.
KNOWN_ROWS, KNOWN_CHARACTERS = 20, 600

SOURCES = {
    "data": "- data: the domain's tables (listed below) - its records, and studies already "
    "computed from them.{studies}",
    "prediction": "- prediction: models that estimate one item the question describes, each "
    "asked with the inputs listed:\n{models}",
    # A how or a why is the documents', whatever the tables call their columns: asked how
    # moisture changes in processing, the plan read a moisture column (2026-10-10).
    "knowledge": "- knowledge: documents that explain how and why, on:\n{topics}\n  A "
    "question about how or why - a process, a cause, a practice - is theirs, even when a "
    "table has a column of the same name.",
}


def sources(models: str, topics: str, studies: str, library: bool) -> str:
    """The sources a plan may use, as the planner is told them: the documents only in a
    domain that has some."""
    shown = ["data", "prediction"] + (["knowledge"] if library else [])
    return "The sources:\n" + "\n".join(
        SOURCES[tool].format(models=models, topics=topics, studies=studies) for tool in shown
    )


def normalised(plan: StepPlan, library: bool) -> StepPlan:
    """The plan as it can run: ids stripped, and in a domain with no documents, a step
    meant for them asked of the tables - the only other source of records."""
    steps = [
        PlannedStep(
            id=step.id.strip(),
            tool="data" if step.tool == "knowledge" and not library else step.tool,
            ask=step.ask.strip(),
            uses=[used.strip() for used in step.uses],
        )
        for step in plan.steps
    ]
    return StepPlan(in_scope=plan.in_scope, steps=steps)


def plan_problems(plan: StepPlan) -> list[str]:
    """What keeps a plan from running, in words the planner can act on; empty when it can."""
    if not plan.in_scope:
        return []
    if not plan.steps:
        return ["A question in scope needs at least one step."]
    found = []
    if len(plan.steps) > MAX_STEPS:
        found.append(f"{len(plan.steps)} steps: plan at most {MAX_STEPS}.")
    earlier: set[str] = set()
    for step in plan.steps:
        if not step.id:
            found.append("Every step needs an id.")
        elif step.id in earlier:
            found.append(f"Two steps are called {step.id}: give each its own id.")
        if not step.ask:
            found.append(f"Step {step.id} asks nothing: write what it finds.")
        for used in step.uses:
            if used == step.id:
                found.append(f"Step {step.id} uses itself.")
            elif used not in earlier:
                found.append(f"Step {step.id} uses {used}, which is not a step planned before it.")
        earlier.add(step.id)
    return found


def waves(steps: Sequence[PlannedStep]) -> list[list[PlannedStep]]:
    """The steps in rounds: each round's steps need only steps of earlier rounds, so they
    can run at once. The plan must be checked first (`plan_problems`): every use is of a
    step planned before."""
    round_of: dict[str, int] = {}
    rounds: list[list[PlannedStep]] = []
    for step in steps:
        number = max((round_of[used] + 1 for used in step.uses), default=0)
        round_of[step.id] = number
        while len(rounds) <= number:
            rounds.append([])
        rounds[number].append(step)
    return rounds


@dataclass(frozen=True)
class StepResult:
    """What one step found - or why it did not run."""

    step: PlannedStep
    sql: SqlAnswer | None = None
    prediction: PredictionAnswer | None = None
    passages: list[dict[str, Any]] = field(default_factory=list)
    skipped: str | None = None  # why it did not run: a step it needs found nothing

    @property
    def tool(self) -> Tool:
        return self.step.tool

    @property
    def found(self) -> bool:
        """Whether it found something a later step, or the answer, can use."""
        if self.skipped is not None:
            return False
        if self.sql is not None:
            return self.sql.result is not None and not self.sql.empty
        if self.prediction is not None:
            return self.prediction.response is not None
        return bool(self.passages)


def known(results: dict[str, StepResult], uses: Sequence[str]) -> str:
    """What the steps a step uses found, as it is handed them: each by its id and ask."""
    if not uses:
        return ""
    blocks = []
    for used in uses:
        result = results[used]
        head = f"[{used}] {result.step.ask}"
        if result.sql is not None and result.sql.result is not None:
            rows = result.sql.result
            cut = len(rows.rows) > KNOWN_ROWS
            shown = replace(rows, rows=rows.rows[:KNOWN_ROWS], truncated=rows.truncated or cut)
            blocks.append(f"{head}\n{shown.as_text()}")
        elif result.prediction is not None and result.prediction.response is not None:
            blocks.append(f"{head}\n{described(result.prediction.response)}")
        elif result.passages:
            blocks.append(f"{head}\n{result.passages[0]['text'][:KNOWN_CHARACTERS]}")
    return "Found by the steps this one uses:\n" + "\n\n".join(blocks)


def route_of(steps: Sequence[PlannedStep]) -> str:
    """The route a plan amounts to: its one tool, or `mixed`."""
    tools = {step.tool for step in steps}
    return tools.pop() if len(tools) == 1 else "mixed"


def from_route_plan(plan: dict[str, str | None]) -> list[PlannedStep]:
    """The router's plan - a part per tool - as steps that need nothing from each other:
    what runs when the planner's own plan cannot."""
    return [
        PlannedStep(id=f"s{n}", tool=tool, ask=part, uses=[])  # type: ignore[arg-type]
        for n, (tool, part) in enumerate(((t, p) for t, p in plan.items() if p), start=1)
    ]
