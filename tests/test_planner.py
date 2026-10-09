"""A question planned as steps: what keeps a plan from running, the rounds it runs in, and
what a step is handed of the steps it needs."""

from mlops_core.agent.planner import (
    MAX_STEPS,
    StepResult,
    from_route_plan,
    known,
    normalised,
    plan_problems,
    route_of,
    sources,
    waves,
)
from mlops_core.agent.prompts import PlannedStep, StepPlan
from mlops_core.agent.sql import QueryResult
from mlops_core.agent.text_to_sql import SqlAnswer
from mlops_core.agent.tools import PredictionAnswer


def step(
    id_: str, tool: str = "data", uses: tuple[str, ...] = (), ask: str = "what?"
) -> PlannedStep:
    return PlannedStep(id=id_, tool=tool, ask=ask, uses=list(uses))  # type: ignore[arg-type]


def plan(*steps: PlannedStep, in_scope: bool = True) -> StepPlan:
    return StepPlan(in_scope=in_scope, steps=list(steps))


def found(rows: list[tuple[object, ...]], columns: tuple[str, ...] = ("hour",)) -> SqlAnswer:
    return SqlAnswer("SELECT 1", QueryResult("SELECT 1", list(columns), rows, False), None, 1)


def test_a_plan_that_can_run_has_no_problems_and_one_out_of_scope_needs_no_steps() -> None:
    assert plan_problems(plan(step("s1"), step("s2", "prediction", ("s1",)))) == []
    assert plan_problems(plan(in_scope=False)) == []
    assert plan_problems(plan()) == ["A question in scope needs at least one step."]


def test_what_keeps_a_plan_from_running_is_said_so_it_can_be_planned_again() -> None:
    problems = plan_problems(
        plan(
            step("s1"),
            step("s1", ask=" "),
            step("s2", uses=("s3",)),
            step("s3", uses=("s3",)),
            step("", "knowledge"),
            step("s5"),
        )
    )

    assert "Two steps are called s1: give each its own id." in problems
    assert "Step s2 uses s3, which is not a step planned before it." in problems
    assert "Step s3 uses itself." in problems
    assert "Every step needs an id." in problems
    assert f"6 steps: plan at most {MAX_STEPS}." in problems
    assert "Step s1 asks nothing: write what it finds." not in problems  # " " is stripped later
    assert plan_problems(normalised(plan(step("s1", ask=" ")), library=True)) == [
        "Step s1 asks nothing: write what it finds."
    ]


def test_without_documents_a_step_for_them_asks_the_tables() -> None:
    planned = plan(step(" s1 ", "knowledge", ask=" Why? "), step("s2", "prediction", (" s1",)))

    assert [s.tool for s in normalised(planned, library=False).steps] == ["data", "prediction"]
    kept = normalised(planned, library=True)
    assert kept.steps[0] == step("s1", "knowledge", ask="Why?")
    assert kept.steps[1].uses == ["s1"]


def test_the_steps_run_in_rounds_each_after_what_it_needs() -> None:
    planned = [
        step("s1"),
        step("s2", "prediction"),
        step("s3", "prediction", ("s1",)),
        step("s4", uses=("s3", "s2")),
    ]

    assert [[s.id for s in wave] for wave in waves(planned)] == [["s1", "s2"], ["s3"], ["s4"]]
    assert waves([]) == []


def test_a_step_is_handed_what_the_steps_it_needs_found_and_nothing_else() -> None:
    many = [(n,) for n in range(30)]
    results = {
        "s1": StepResult(step("s1", ask="Busiest hour?"), sql=found([(10,)])),
        "s2": StepResult(
            step("s2", "prediction", ask="Tickets then?"),
            prediction=PredictionAnswer(
                "demand",
                {},
                {
                    "target": "tickets",
                    "prediction": 12.5,
                    "model_version": "3",
                    "model_source": "registry",
                    "context": {},
                },
                None,
            ),
        ),
        "s3": StepResult(step("s3", "knowledge"), passages=[{"text": "x" * 900}]),
        "s4": StepResult(step("s4", ask="Every hour?"), sql=found(many)),
        "s5": StepResult(step("s5"), skipped="it needs s9"),
    }

    handed = known(results, ["s1", "s2", "s3", "s4", "s5"])

    assert handed.startswith("Found by the steps this one uses:\n[s1] Busiest hour?\nhour\n10")
    assert "[s2] Tickets then?\n" in handed and "12.5" in handed
    assert "x" * 600 in handed and "x" * 601 not in handed
    assert "(only the first 20 rows)" in handed and "\n29" not in handed
    assert "[s5]" not in handed  # it found nothing to hand on
    assert known(results, []) == ""


def test_what_a_step_found_is_what_a_later_one_or_the_answer_can_use() -> None:
    empty = SqlAnswer("SELECT 1", QueryResult("SELECT 1", ["n"], [], False), None, 1)
    failed = SqlAnswer("SELECT", None, "Parser Error", 2)

    assert StepResult(step("s1"), sql=found([(1,)])).found
    assert not StepResult(step("s1"), sql=empty).found
    assert not StepResult(step("s1"), sql=failed).found
    assert not StepResult(step("s1"), sql=found([(1,)]), skipped="no").found
    assert not StepResult(
        step("s1", "prediction"), prediction=PredictionAnswer("m", {}, None, "down")
    ).found
    assert StepResult(step("s1", "knowledge"), passages=[{"text": "a"}]).found
    assert not StepResult(step("s1", "knowledge")).found
    assert StepResult(step("s1", "knowledge")).tool == "knowledge"


def test_a_plan_amounts_to_a_route_and_the_routers_plan_to_steps() -> None:
    assert route_of([step("s1"), step("s2")]) == "data"
    assert route_of([step("s1"), step("s2", "prediction")]) == "mixed"
    assert from_route_plan({"data": "How many?", "prediction": None, "knowledge": "Why?"}) == [
        step("s1", ask="How many?"),
        step("s2", "knowledge", ask="Why?"),
    ]


def test_the_planner_is_told_only_the_sources_the_domain_has() -> None:
    told = sources("  - demand: tickets", "  - roasting: how", " Studies: x.", library=True)

    assert told.startswith("The sources:\n- data: the domain's tables")
    assert "studies already computed from them. Studies: x." in told
    assert "  - demand: tickets" in told and "  - roasting: how" in told
    assert "knowledge" not in sources("  - demand: tickets", "", "", library=False)
