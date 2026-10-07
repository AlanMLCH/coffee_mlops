"""The agent's SQL: read-only by construction, bounded in rows, time and memory.

The guardrails live in the database session and in a parser, never in the prompt: a
prompt is a request, and the text the agent reads - a shop's description, a document -
can carry instructions of its own. Verified on DuckDB 1.5.5 (2026-09-25):

- With external access off, the allowed directories narrowed to the published layers and
  the configuration locked, the catalog's views still read, while files anywhere else
  (the raw layer included), URLs, extension installs and setting changes are refused.
- But `COPY ... TO` into an allowed directory still writes, and CREATE and DROP run: the
  settings alone do not make a session read-only. So every statement is parsed first,
  and anything but a single SELECT is refused before the engine sees it.
- DuckDB has no statement timeout; a timer interrupts the query and the session stays
  usable. Results stream, so fetching a few rows of a huge result costs a few rows.
- One connection runs one query at a time, and the explorer's tabs, the agent and the MCP
  server's clients ask at once. Every query runs on its own cursor - another connection
  to the same database - which keeps the session's settings: verified 2026-09-29, a
  cursor refuses the raw layer, URLs, installs and setting changes like the session does,
  four cursors run side by side, and interrupting one stops only its query. So no lock.
"""

import re
import threading
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from mlops_core.catalog import LAYERS, connect
from mlops_core.config import DATA_LAYERS, ParentTables
from mlops_core.storage import table_path

MAX_ROWS = 50  # what an answer can use; more is a sign the query should aggregate
TIMEOUT_SECONDS = 10.0
MEMORY_LIMIT = "1GB"


class Refused(ValueError):
    """A statement the agent may not run, said the way a model can correct it."""


@dataclass(frozen=True)
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool  # more rows than MAX_ROWS matched

    def as_text(self) -> str:
        """Pipe-separated rows under a header: what a model reads back."""
        lines = [" | ".join(self.columns)]
        # A value's own line breaks (a shop's description) would read as more rows.
        lines += [" | ".join("" if v is None else " ".join(str(v).split()) for v in row)
                  for row in self.rows]  # fmt: skip
        if self.truncated:
            lines.append(f"(only the first {len(self.rows)} rows)")
        return "\n".join(lines)


def read_only(data_dir: Path, parent: ParentTables | None = None) -> duckdb.DuckDBPyConnection:
    """A session over the catalog's views that can read the published layers - and, for a
    subdomain, the parent's tables it lists - and nothing else, and cannot be talked into
    changing that. A domain's own layers never hold its subdomains' data, so its session
    reads none of theirs."""
    root = data_dir.resolve()
    con = connect(root, parent)
    allowed = [f"{(root / layer).as_posix()}/" for layer in LAYERS if (root / layer).is_dir()]
    lent = [table_path(root, name) for name in (parent.qualified() if parent else [])]
    allowed += [f"{path.as_posix()}/" for path in lent if path.is_dir()]
    listing = ", ".join("'" + path.replace("'", "''") + "'" for path in allowed)
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET allowed_directories = [{listing}]")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def run_select(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    max_rows: int = MAX_ROWS,
    timeout: float = TIMEOUT_SECONDS,
) -> QueryResult:
    """Run one SELECT on a cursor of its own and return at most `max_rows` rows. Raises
    `Refused` for anything else, and DuckDB's own error - a wrong column, a timeout - for
    the model to read. Safe to call from several threads at once."""
    with con.cursor() as cursor:
        statements = cursor.extract_statements(sql)
        if len(statements) != 1:
            raise Refused(f"Send exactly one statement; this is {len(statements)}")
        if statements[0].type != duckdb.StatementType.SELECT:
            raise Refused(f"Only SELECT may run; this is {statements[0].type.name}")
        query = statements[0].query.strip()
        timer = threading.Timer(timeout, cursor.interrupt)
        timer.start()
        try:
            rows = cursor.execute(query).fetchmany(max_rows + 1)
        finally:
            timer.cancel()
        columns = [column[0] for column in cursor.description or []]
    return QueryResult(query, columns, rows[:max_rows], len(rows) > max_rows)


_QUOTED = re.compile(r'"(\w+\.\w+)"')


def unquoted_views(sql: str, names: Collection[str]) -> str:
    """`"clean.shops"` -> `clean.shops`, for the views the session has.

    A small model often quotes a view's whole name as one identifier, which DuckDB reads
    as a table called "clean.shops" in no schema, and its repairs keep the quotes (seen
    2026-09-28: the same question failed that way on one run and with a parser error on
    the next). Only a quoted name that is exactly one of `names` is touched, so a quoted
    column or alias with a dot in it is left alone.
    """
    return _QUOTED.sub(lambda m: m.group(1) if m.group(1) in names else m.group(0), sql)


_NAMED = re.compile(rf"(?<![\w.])((?:\w+\.)?(?:{'|'.join(DATA_LAYERS)})\.\w+)(?![\w.])")


def qualified_views(sql: str, names: Collection[str]) -> str:
    """A view named with the wrong domain or layer, named as the session has it - when
    the session has no view of the name as written and exactly one of that table:
    `clean.shops` -> `market.clean.shops` (a parent's table written without its domain),
    `clean.alerts` -> `analysis.alerts` (a table put in another layer).

    Seen 2026-10-05: a subdomain's agent wrote two of its parent's tables without the
    parent's name, and every repair kept them so; and 2026-10-07, two of its own studies
    under `clean.` and under its parent's name. A name the session has, or one that could
    be either of two tables, is left as it is, for the error to say so.
    """

    def qualify(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in names:
            return name
        table = name.rsplit(".", 1)[1]
        found = [view for view in names if view.rsplit(".", 1)[1] == table]
        return found[0] if len(found) == 1 else name

    return _NAMED.sub(qualify, sql)


def views(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Every `layer.table` the session can query, and every `<domain>.<layer>.<table>`
    another domain lends it."""
    with con.cursor() as cursor:
        found = cursor.execute(
            "SELECT CASE WHEN table_catalog = current_database() THEN '' "
            "ELSE table_catalog || '.' END || table_schema || '.' || table_name "
            "FROM information_schema.tables"
        ).fetchall()
    return {str(name) for (name,) in found}
