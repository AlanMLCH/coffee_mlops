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
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from mlops_core.catalog import LAYERS, connect

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
        lines += [" | ".join("" if v is None else str(v) for v in row) for row in self.rows]
        if self.truncated:
            lines.append(f"(only the first {len(self.rows)} rows)")
        return "\n".join(lines)


def read_only(data_dir: Path) -> duckdb.DuckDBPyConnection:
    """A session over the catalog's views that can read the published layers and
    nothing else, and cannot be talked into changing that."""
    root = data_dir.resolve()
    con = connect(root)
    allowed = [f"{(root / layer).as_posix()}/" for layer in LAYERS if (root / layer).is_dir()]
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
    """Run one SELECT and return at most `max_rows` rows. Raises `Refused` for anything
    else, and DuckDB's own error - a wrong column, a timeout - for the model to read."""
    statements = con.extract_statements(sql)
    if len(statements) != 1:
        raise Refused(f"Send exactly one statement; this is {len(statements)}")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise Refused(f"Only SELECT may run; this is {statements[0].type.name}")
    query = statements[0].query.strip()
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        cursor = con.execute(query)
        rows = cursor.fetchmany(max_rows + 1)
    finally:
        timer.cancel()
    columns = [column[0] for column in cursor.description or []]
    return QueryResult(query, columns, rows[:max_rows], len(rows) > max_rows)


def views(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Every `layer.table` the session can query."""
    found = con.execute(
        "SELECT table_schema || '.' || table_name FROM information_schema.tables"
    ).fetchall()
    return {str(name) for (name,) in found}
