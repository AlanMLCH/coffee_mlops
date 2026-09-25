"""The domain's data dictionary, as the schema the agent's SQL is written against.

A Markdown file beside the domain's code, kept honest by a test that fails when a column
of a contract goes undocumented; one section per table, headed with the view it
describes (`## `layer.table` - what one row is`). The model is shown the sections of the
tables that exist, verbatim: units, nulls and traps ("null means not reported, never
zero") are exactly what a model writing SQL gets wrong unless it is told.
"""

import re
from collections.abc import Collection
from itertools import pairwise
from pathlib import Path

DICTIONARY_FILE = "data_dictionary.md"

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


def schema_context(text: str, views: Collection[str]) -> str:
    """The dictionary's sections for the tables that exist as views, in its own order:
    a table the pipeline has not built yet is not offered to the model."""
    return "\n\n".join(section for view, section in table_sections(text).items() if view in views)
