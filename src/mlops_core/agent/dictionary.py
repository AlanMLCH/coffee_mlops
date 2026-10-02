"""The domain's data dictionary, as the schema the agent's SQL is written against.

A Markdown file beside the domain's code, kept honest by a test that fails when a column
of a contract goes undocumented; one section per table, headed with the view it
describes (`## `layer.table` - what one row is`). The model is shown the sections of the
tables that exist, verbatim: units, nulls and traps ("null means not reported, never
zero") are exactly what a model writing SQL gets wrong unless it is told.
"""

import re
from collections.abc import Callable, Collection, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

DICTIONARY_FILE = "data_dictionary.md"
# The models' inputs are not offered. Everything in them comes from the clean layer, some
# of it shifted on purpose - an item's context is the period before its own - and a
# question read against them gets the shifted figure for the fact. Measured (2026-09-25):
# asked for a country's figure in one year, the agent read one item's lagged context and
# answered with it.
MODEL_INPUTS = "features."

_SECTION = re.compile(r"^## ", re.MULTILINE)
_VIEW = re.compile(r"^## `(\w+\.\w+)`")


def dictionary_path(domain_dir: Path) -> Path:
    return domain_dir / DICTIONARY_FILE


def table_sections(text: str) -> dict[str, str]:
    """`layer.table` -> its section, heading included, for every section that names a
    view; sections about anything else (the raw layer) are left out."""
    starts = [m.start() for m in _SECTION.finditer(text)] + [len(text)]
    sections = {}
    for start, end in pairwise(starts):
        section = text[start:end].strip()
        view = _VIEW.match(section)
        if view:
            sections[view[1]] = section
    return sections


def offered(text: str, views: Collection[str]) -> dict[str, str]:
    """The sections of the tables the agent answers from: those that exist as views - a
    table the pipeline has not built yet is not offered - and are not a model's inputs."""
    return {
        view: section
        for view, section in table_sections(text).items()
        if view in views and not view.startswith(MODEL_INPUTS)
    }


def shown(views: Collection[str], hidden: Collection[str]) -> set[str]:
    """The views the agent is shown: all of them but those the domain keeps from it."""
    return set(views) - set(hidden)


def reads_any(sql: str, tables: Collection[str]) -> bool:
    """Whether a query names one of these tables: a case the agent cannot answer when
    they are kept from it."""
    return any(re.search(rf"\b{re.escape(table)}\b", sql, re.IGNORECASE) for table in tables)


def schema_context(text: str, views: Collection[str]) -> str:
    """The offered tables' sections, in the dictionary's own order."""
    return "\n\n".join(offered(text, views).values())


# What a section is matched on: its heading and opening, where it says what a row is.
# The query-sized embedding context (512 tokens) would cut a whole section anyway.
SUMMARY_CHARS = 1200


class SchemaLinker:
    """The sections a question needs, instead of all of them: the `k` most similar to it,
    and the ones those name (the other side of a join they describe), in the dictionary's
    order.

    The whole dictionary is most of the SQL writer's prompt (about 8,000 tokens by
    September 2026), and a small model reads past what it was told in a long prompt.
    Schema linking is the usual remedy in text to SQL; whether it helps here is measured,
    not assumed (`agent.schema_sections`).
    """

    def __init__(
        self,
        sections: dict[str, str],
        embed: Callable[[Sequence[str]], np.ndarray],
        k: int,
        query: Callable[[str], str] = lambda question: question,
    ):
        self._sections = sections
        self._names = list(sections)
        self._embed, self._k, self._query = embed, k, query
        self._vectors = embed([section[:SUMMARY_CHARS] for section in sections.values()])

    def chosen(self, question: str) -> list[str]:
        similarity = self._vectors @ self._embed([self._query(question)])[0]
        top = {self._names[i] for i in np.argsort(-similarity)[: self._k]}
        named = named_by(self._sections, top)
        return [name for name in self._names if name in top | named]

    def __call__(self, question: str) -> str:
        return "\n\n".join(self._sections[name] for name in self.chosen(question))


def named_by(sections: dict[str, str], chosen: Collection[str]) -> set[str]:
    """The sections the chosen ones name and are not among them: the other side of a join
    a chosen section describes."""
    return {
        name
        for name in sections
        if name not in chosen and any(f"`{name}`" in sections[t] for t in chosen)
    }
