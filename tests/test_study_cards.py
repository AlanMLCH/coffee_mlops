"""The studies a question is shown: cards from the dictionary, found by retrieval."""

from collections.abc import Sequence

import numpy as np

from mlops_core.agent.prompts import NEEDS, PLAN, ROUTER
from mlops_core.agent.study_cards import NO_STUDIES, StudyFinder, study_cards

SECTIONS = {
    "clean.sales": "## `clean.sales` — one line of a ticket\n\nA product sold.\n\n| a | b |",
    "analysis.scenarios": (
        "## `analysis.scenarios` — a day now, and with every price moved\n\n"
        "Tickets move by the elasticity,\ntaken as constant.\n\n| Column | Type |\n|---|---|\n"
        "| `change_pct` | Float |\n| `tickets_per_day`, `revenue_per_day` | Float |"
    ),
    "analysis.staffing": "## `analysis.staffing` — each slot: its crowd and its hands\n| x | y |",
    "market.analysis.outlook": "## `market.analysis.outlook`\n\nWhere a price may go.\n",
}


def embed(texts: Sequence[str]) -> np.ndarray:
    """Two directions: prices, and staff."""
    return np.array(
        [[float("price" in t), float("staff" in t or "crowd" in t or "hands" in t)] for t in texts]
    )


def test_a_study_is_an_analysis_table_this_domains_or_its_parents() -> None:
    cards = study_cards(SECTIONS)

    assert [card.view for card in cards] == [
        "analysis.scenarios",
        "analysis.staffing",
        "market.analysis.outlook",
    ]
    scenarios, staffing, outlook = cards
    assert scenarios.title == "a day now, and with every price moved"
    assert scenarios.summary == "Tickets move by the elasticity, taken as constant."
    assert staffing.summary == ""  # no paragraph before its table
    assert outlook.title == "market.analysis.outlook"  # a heading with no title: its name
    assert outlook.summary == "Where a price may go."
    assert scenarios.columns == ("change_pct", "tickets_per_day", "revenue_per_day")
    assert staffing.columns == () and outlook.columns == ()
    assert scenarios.listing() == (
        "    - analysis.scenarios: a day now, and with every price moved."
        " Holds: change_pct, tickets_per_day, revenue_per_day."
    )
    assert "tickets per day" in scenarios.text


def test_the_closest_studies_are_shown_to_every_step_that_chooses_a_tool() -> None:
    finder = StudyFinder(study_cards(SECTIONS), embed, shown=1)

    assert [card.view for card in finder.closest("What if every price rose?")] == [
        "analysis.scenarios"
    ]
    blocks = finder.blocks("Is my staff enough for the crowd?")
    assert "analysis.staffing" in blocks["studies"] and "scenarios" not in blocks["studies"]
    assert "analysis.staffing" in blocks["needs_studies"]
    assert "analysis.staffing" in blocks["plan_studies"]

    router = ROUTER.format(question="q", subject="s", tables="", models="", topics="", **blocks)
    assert "A question one of them answers is data" in router
    needs = NEEDS.format(subject="s", models="  - m: one", question="q", **blocks)
    assert "analysis.staffing: each slot" in needs and "\n  The models, each" in needs


def test_a_domain_with_no_studies_reads_the_prompts_as_before() -> None:
    finder = StudyFinder(study_cards({"clean.sales": SECTIONS["clean.sales"]}), embed, shown=2)

    assert finder.closest("anything") == [] and finder.blocks("anything") == NO_STUDIES
    router = ROUTER.format(question="q", subject="s", tables="", models="", topics="", **NO_STUDIES)
    assert "(listed\n  below).\n- prediction:" in router and "studies" not in router
    plan = PLAN.format(subject="s", models="", question="q", **NO_STUDIES)
    assert "of what they list.\n- prediction:" in plan
