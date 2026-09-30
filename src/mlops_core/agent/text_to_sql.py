"""Text to SQL: the model writes a query, the locked-down session runs it, and what went
wrong goes back to the model - each kind of trouble a bounded number of times.

- An error goes back for repair, twice at most.
- A query that ran but ignored a rule the domain declares for a table it reads
  (`agent.sql_guards`) goes back once, with the rule.
- A query that ran and found nothing goes back once, with the values it filtered on
  checked against the data: a filter that names a value the data spells otherwise finds
  nothing, and nothing is not an answer. The rewrite is kept only if each value it
  changed is a spelling of the same one - shown the shops there are, the model once put
  another shop in the place of one the data does not have. If nothing is found, that is
  what the agent reports - the answer step then has nothing to invent a figure from.
- A query that reads no table is refused, like an error: a figure it returns is the
  model's, not the data's (asked who won a World Cup, it wrote `SELECT 'Brazil'`).

A query that runs and answers another question looks, from inside the agent, like a
right one: that is what the benchmark exists to measure.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

import duckdb
from pydantic import BaseModel

from mlops_core.agent.prompts import EMPTY, GUARD, REPAIR, SQL, SqlReply
from mlops_core.agent.sql import (
    MAX_ROWS,
    QueryResult,
    Refused,
    run_select,
    unquoted_views,
    views,
)
from mlops_core.config import SqlGuard

MAX_ATTEMPTS = 3  # the first query and two repairs
SHOWN_VALUES = 12  # of a column whose filter value matched nothing
# How alike two spellings of one value are, at least: "São Paulo" and "sao paulo",
# "Zurich" and "Zürich" pass; "acme" and "apex" do not.
SAME_VALUE = 0.8
NO_TABLE = "The query reads no table: every figure must come from the tables described."
# `column = 'value'`, `column ILIKE 'value'` or `column IN ('a', 'b')`, qualified or not.
_COMPARED = re.compile(r"(?:\w+\.)?(\w+)\s*(?:=|I?LIKE)\s*'((?:[^']|'')*)'", re.IGNORECASE)
_LISTED = re.compile(r"(?:\w+\.)?(\w+)\s+IN\s*\(([^)]*)\)", re.IGNORECASE)
_QUOTED = re.compile(r"'((?:[^']|'')*)'")


class Generator(Protocol):
    """A model that answers a prompt with a reply of the given shape."""

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply: ...


@dataclass(frozen=True)
class SqlAnswer:
    sql: str  # the last query written
    result: QueryResult | None  # None when every attempt failed
    error: str | None  # the last error, when every attempt failed
    attempts: int

    @property
    def empty(self) -> bool:
        """The query ran and found nothing: no rows, or rows of nothing but nulls."""
        return self.result is not None and nothing_in(self.result)


def nothing_in(result: QueryResult) -> bool:
    return not result.rows or all(value is None for row in result.rows for value in row)


def write_sql(
    generator: Generator,
    con: duckdb.DuckDBPyConnection,
    schema: str,
    question: str,
    max_rows: int = MAX_ROWS,
    guards: Sequence[SqlGuard] = (),
) -> SqlAnswer:
    """Ask for a query and run it; repair what failed, then check what ran against the
    domain's guards and, if it found nothing, against the data. A view's whole name quoted
    as one identifier is unquoted first (`sql.unquoted_views`)."""
    prompt = SQL.format(schema=schema, question=question)
    names = views(con)
    answer = _repaired(generator, con, prompt, names, max_rows)
    if answer.result is None:
        return answer
    for guard in guards:
        if ignores(answer.sql, guard):
            hint = GUARD.format(sql=answer.sql, hint=guard.hint)
            answer = _again(generator, con, prompt + hint, names, max_rows, answer)
            break
    if answer.empty:
        absent = "\n".join(absent_values(con, answer.sql, names))
        again = EMPTY.format(sql=answer.sql, absent=absent)
        rewritten = _again(generator, con, prompt + again, names, max_rows, answer)
        answer = rewritten if same_values(answer.sql, rewritten.sql) else answer
    return answer


def same_values(before: str, after: str) -> bool:
    """Whether every text value the rewrite compares a column with is one the first query
    compared that column with, spelled alike: a rewrite may fix a spelling, never swap in
    another value."""
    first: dict[str, list[str]] = {}
    for column, value in _compared(before):
        first.setdefault(column.lower(), []).append(_folded(value))
    for column, value in _compared(after):
        earlier = first.get(column.lower())
        if earlier is not None and not any(
            SequenceMatcher(None, _folded(value), old).ratio() >= SAME_VALUE for old in earlier
        ):
            return False
    return True


def absent_values(con: duckdb.DuckDBPyConnection, sql: str, names: set[str]) -> list[str]:
    """For each text value the query compares a column with, where no row of the tables it
    reads has it: which values that column does have. What a model filtered on is often
    what the data spells another way ("Acme", "ACME Inc.", or nothing at all)."""
    compared = _compared(sql)
    read = [n for n in sorted(names) if re.search(rf"\b{re.escape(n)}\b", sql, re.IGNORECASE)]
    lines = []
    for column, value in compared:
        for table in read:
            if column not in _columns(con, table):
                continue
            # The literal as the query wrote it: a quote inside it is already doubled.
            found = _scalar(
                con,
                f"SELECT count(*) FROM {table} WHERE lower(CAST({column} AS VARCHAR)) = "
                f"lower('{value}')",
            )
            if found:
                continue
            values = run_select(
                con,
                f"SELECT DISTINCT CAST({column} AS VARCHAR) FROM {table} "
                f"WHERE {column} IS NOT NULL ORDER BY 1 LIMIT {SHOWN_VALUES}",
            ).rows
            shown = ", ".join(str(row[0]) for row in values)
            lines.append(
                f"{column} = '{value}' matches no row of {table}; its values include: {shown}"
            )
    return lines


def _compared(sql: str) -> list[tuple[str, str]]:
    """Each (column, text value) the query compares: `=`, `LIKE`, `ILIKE` and `IN`."""
    compared = [(column, value) for column, value in _COMPARED.findall(sql)]
    for column, listed in _LISTED.findall(sql):
        compared += [(column, value) for value in _QUOTED.findall(listed)]
    return compared


def _folded(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.replace("''", "'"))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def _reads_a_table(sql: str, names: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(name)}\b", sql, re.IGNORECASE) for name in names)


def _repaired(
    generator: Generator,
    con: duckdb.DuckDBPyConnection,
    prompt: str,
    names: set[str],
    max_rows: int,
) -> SqlAnswer:
    """The first query that runs, an error handed back for repair each time."""
    sql, error = "", ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        repair = REPAIR.format(sql=sql, error=error) if attempt > 1 else ""
        sql = unquoted_views(generator.ask(prompt + repair, SqlReply).sql.strip(), names)
        if not _reads_a_table(sql, names):
            error = NO_TABLE
            continue
        try:
            return SqlAnswer(sql, run_select(con, sql, max_rows), None, attempt)
        except (Refused, duckdb.Error) as failed:
            # The first line is the error; the rest is DuckDB's context, noise to a model.
            error = str(failed).strip().splitlines()[0]
    return SqlAnswer(sql, None, error, MAX_ATTEMPTS)


def _again(
    generator: Generator,
    con: duckdb.DuckDBPyConnection,
    prompt: str,
    names: set[str],
    max_rows: int,
    before: SqlAnswer,
) -> SqlAnswer:
    """One more query; kept only if it runs - a rewrite that fails loses nothing."""
    sql = unquoted_views(generator.ask(prompt, SqlReply).sql.strip(), names)
    kept = SqlAnswer(before.sql, before.result, None, before.attempts + 1)
    if not _reads_a_table(sql, names):
        return kept
    try:
        return SqlAnswer(sql, run_select(con, sql, max_rows), None, before.attempts + 1)
    except (Refused, duckdb.Error):
        return kept


def ignores(sql: str, guard: SqlGuard) -> bool:
    """The query reads the guarded table and never names the column it must filter on.
    Public: the MCP server tells its clients the same thing."""
    text = sql.lower()
    reads = re.search(rf"\b{re.escape(guard.table.lower())}\b", text) is not None
    return reads and re.search(rf"\b{re.escape(guard.requires.lower())}\b", text) is None


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    schema, name = table.split(".", 1)
    rows = run_select(
        con,
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{name}'",
    ).rows
    return {row[0] for row in rows}


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> object:
    return run_select(con, sql).rows[0][0]
