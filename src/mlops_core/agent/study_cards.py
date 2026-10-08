"""Which studies a question may be asking for, found by retrieval, as the models are.

A domain's studies (its `analysis` tables, and those its parent lends) are answers already
computed from the records: what a change would do to the whole business, how it answers a
price, where a price may go. Read as one more heading in the list of tables, they lost to
the models: a question about what would happen sounds like a prediction, and the router,
the second opinion and every generator tried (a 4B, Gemini Flash-Lite, Groq's 120B) sent
"what would a 10% rise do to a day's gross profit" to a model that predicts one hour at a
time - and once a raw material's price outlook to a model of hourly sales (2026-10-07).
So each study
gets a card - its heading and what its section says first - embedded once; a question is
set against the cards, and the steps that choose a tool are shown the closest few, as
answers already there.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mlops_core.agent.model_cards import Embed

STUDIES = "analysis"  # the layer a domain's studies are kept in
# How a question is embedded for this search: a task of its own (qwen3-embedding takes it
# as an instruction), as the models' is.
STUDY_TASK = "Given a question, retrieve the study whose computed result answers it"
_HEADING = re.compile(r"^## `(?P<view>[^`]+)`(?: — (?P<title>.*))?$")
_NAMED = re.compile(r"`(\w+)`")


@dataclass(frozen=True)
class StudyCard:
    """A study as the tool choice is shown it: its view, its title, its first words,
    and the columns it holds - what a model card's inputs are to a model. Shown only
    its title ("a day now, and with every price moved"), the price scenarios lost to the
    model whose card says "with every price moved by a percent" (2026-10-07)."""

    view: str
    title: str
    summary: str  # the section's first paragraph, if it has one before its table
    columns: tuple[str, ...] = ()  # as its table of columns names them

    def listing(self) -> str:
        """The card as a prompt lists it."""
        holds = f" Holds: {', '.join(self.columns)}." if self.columns else ""
        return f"    - {self.view}: {self.title}.{holds}"

    @property
    def text(self) -> str:
        """What is embedded: the title, the first words, and the columns as words."""
        columns = ", ".join(column.replace("_", " ") for column in self.columns)
        return f"{self.title}. {self.summary} {columns}".strip()


def study_cards(sections: Mapping[str, str]) -> list[StudyCard]:
    """A card per study among the offered sections - a view in the studies' layer, this
    domain's (`analysis.x`) or a parent's (`<parent>.analysis.x`) - in their order."""
    cards = []
    for view, section in sections.items():
        if view.split(".")[-2] != STUDIES:
            continue
        lines = section.splitlines()
        heading = _HEADING.match(lines[0])
        title = (heading.group("title") if heading else None) or view
        paragraph: list[str] = []
        for line in lines[1:]:
            if line.startswith("|"):
                break
            if line.strip():
                paragraph.append(line.strip())
            elif paragraph:
                break
        # The names in the first cell of each row of its table of columns, past the
        # header and its rule.
        cells = [
            line.split("|")[1] for line in lines if line.startswith("|") and line.count("|") > 2
        ]
        columns = tuple(name for cell in cells[2:] for name in _NAMED.findall(cell))
        cards.append(StudyCard(view, title, " ".join(paragraph), columns))
    return cards


def study_query(question: str) -> str:
    return f"Instruct: {STUDY_TASK}\nQuery:{question}"


# What the steps that choose a tool are told, with the closest studies in it. Each block
# is empty when the domain shows no studies, so its prompts read as they always did.
ROUTER_STUDIES = """ Among them are studies, answers already computed from the
  records for the whole business. The studies closest to the question:
{cards}
  A question one of them answers is data, even when it asks what would happen or what to
  expect: the study holds the answer. A model is for one item the question describes."""
NEEDS_STUDIES = """ Nor is a question one of these studies - answers already computed for
  the whole business - answers:
{cards}
 """
NO_STUDIES = {"studies": "", "needs_studies": "", "plan_studies": ""}
PLAN_STUDIES = """ Or what one of these studies, already computed
  for the whole business, holds:
{cards}"""


class StudyFinder:
    """The study cards closest to a question, most similar first."""

    def __init__(self, cards: Sequence[StudyCard], embed: Embed, shown: int):
        self.cards, self._embed, self._shown = list(cards), embed, shown
        self._vectors = embed([card.text for card in self.cards]) if self.cards else None

    def closest(self, question: str) -> list[StudyCard]:
        if self._vectors is None:
            return []
        similarity = self._vectors @ self._embed([study_query(question)])[0]
        return [self.cards[i] for i in np.argsort(-similarity)[: self._shown]]

    def blocks(self, question: str) -> dict[str, str]:
        """What the router, the second opinion and the plan are told about the studies
        closest to `question`; empty strings when there are none."""
        cards = "\n".join(card.listing() for card in self.closest(question))
        if not cards:
            return dict(NO_STUDIES)
        return {
            "studies": ROUTER_STUDIES.format(cards=cards),
            "needs_studies": NEEDS_STUDIES.format(cards=cards),
            "plan_studies": PLAN_STUDIES.format(cards=cards),
        }
