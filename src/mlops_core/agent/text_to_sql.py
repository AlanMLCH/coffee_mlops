"""Text to SQL: the model writes a query, the locked-down session runs it, and an error
goes back to the model to repair - twice at most.

The loop only repairs what fails. A query that runs and answers the wrong question looks,
from inside the agent, like a right one: that is what the benchmark exists to measure.
"""

from dataclasses import dataclass
from typing import Protocol

import duckdb
from pydantic import BaseModel

from mlops_core.agent.prompts import REPAIR, SQL, SqlReply
from mlops_core.agent.sql import MAX_ROWS, QueryResult, Refused, run_select

MAX_ATTEMPTS = 3  # the first query and two repairs


class Generator(Protocol):
    """A model that answers a prompt with a reply of the given shape."""

    def ask[Reply: BaseModel](self, prompt: str, reply: type[Reply]) -> Reply: ...


@dataclass(frozen=True)
class SqlAnswer:
    sql: str  # the last query written
    result: QueryResult | None  # None when every attempt failed
    error: str | None  # the last error, when every attempt failed
    attempts: int


def write_sql(
    generator: Generator,
    con: duckdb.DuckDBPyConnection,
    schema: str,
    question: str,
    max_rows: int = MAX_ROWS,
) -> SqlAnswer:
    """Ask for a query, run it, and hand an error back for repair until one runs."""
    prompt = SQL.format(schema=schema, question=question)
    sql, error = "", ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        repair = REPAIR.format(sql=sql, error=error) if attempt > 1 else ""
        sql = generator.ask(prompt + repair, SqlReply).sql.strip()
        try:
            return SqlAnswer(sql, run_select(con, sql, max_rows), None, attempt)
        except (Refused, duckdb.Error) as failed:
            # The first line is the error; the rest is DuckDB's context, noise to a model.
            error = str(failed).strip().splitlines()[0]
    return SqlAnswer(sql, None, error, MAX_ATTEMPTS)
