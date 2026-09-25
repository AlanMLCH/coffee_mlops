"""The agent's SQL session and the schema it is shown.

Every refusal below was first seen to succeed without its guard: DuckDB with external
access off still let `COPY ... TO` write inside an allowed directory, which is why the
parser check exists at all.
"""

from pathlib import Path

import duckdb
import polars as pl
import pytest

from mlops_core.adapter import domain_dir
from mlops_core.agent.dictionary import dictionary_path, schema_context, table_sections
from mlops_core.agent.sql import Refused, read_only, run_select, views
from mlops_core.storage import write_table


@pytest.fixture
def session(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """A published clean table and, beside it, a raw file the agent must not reach."""
    data = tmp_path / "coffee"
    write_table(pl.DataFrame({"country": ["A", "B", "C"], "points": [80.0, 85.5, 90.25]}),
                data / "clean" / "lots", {})  # fmt: skip
    (data / "raw").mkdir()
    (data / "raw" / "secret.csv").write_text("token\nabc\n", encoding="utf-8")
    return read_only(data)


def test_a_select_runs_and_says_when_it_was_cut(session: duckdb.DuckDBPyConnection) -> None:
    result = run_select(session, "SELECT country, points FROM clean.lots ORDER BY points DESC", 2)

    assert result.columns == ["country", "points"]
    assert result.rows == [("C", 90.25), ("B", 85.5)]
    assert result.truncated
    assert result.as_text().splitlines() == [
        "country | points",
        "C | 90.25",
        "B | 85.5",
        "(only the first 2 rows)",
    ]


@pytest.mark.parametrize(
    ("sql", "refusal"),
    [
        ("COPY (SELECT 1) TO 'clean/leak.csv'", "Only SELECT may run; this is COPY"),
        ("DROP VIEW clean.lots", "this is DROP"),
        ("CREATE TABLE t AS SELECT 1", "this is CREATE"),
        ("SET enable_external_access = true", "this is SET"),
        ("SELECT 1; SELECT 2", "exactly one statement"),
    ],
)
def test_anything_but_one_select_is_refused_before_it_runs(
    session: duckdb.DuckDBPyConnection, sql: str, refusal: str
) -> None:
    with pytest.raises(Refused, match=refusal):
        run_select(session, sql)

    assert run_select(session, "SELECT count(*) FROM clean.lots").rows == [(3,)]


def test_the_session_reads_the_layers_and_nothing_else(
    session: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    raw = (tmp_path / "coffee" / "raw" / "secret.csv").as_posix()

    with pytest.raises(duckdb.PermissionException):
        run_select(session, f"SELECT * FROM read_csv('{raw}')")
    with pytest.raises(duckdb.PermissionException):
        run_select(session, "SELECT * FROM read_csv('https://example.com/x.csv')")
    # Locked: not even the connection's owner can reopen it.
    with pytest.raises(duckdb.InvalidInputException, match="configuration"):
        session.execute("SET enable_external_access = true")


def test_a_query_that_runs_too_long_is_interrupted_and_the_session_survives(
    session: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(duckdb.InterruptException):
        run_select(session, "SELECT count(*) FROM range(1000000000000) a", timeout=0.2)

    assert run_select(session, "SELECT 1 AS one").rows == [(1,)]
    assert views(session) == {"clean.lots"}


# --- The schema the model is shown ---------------------------------------------------------

DICTIONARY = """# Data dictionary

## `clean.lots` — one lot

| Column | Type | Meaning |
|---|---|---|
| `points` | Float | Cup score |

## `features.lot_features` — model input

Built later.

## Raw layer

Nothing here is a view.
"""


def test_only_sections_about_a_view_that_exists_are_shown() -> None:
    assert list(table_sections(DICTIONARY)) == ["clean.lots", "features.lot_features"]
    shown = schema_context(DICTIONARY, {"clean.lots"})

    assert shown.startswith("## `clean.lots` — one lot")
    assert "`points`" in shown
    assert "lot_features" not in shown
    assert "Raw layer" not in shown


def test_a_models_inputs_are_not_offered_even_when_built() -> None:
    """A feature can be a fact shifted on purpose (last year's market for this year's
    lot): read as the fact, it answers the wrong year."""
    shown = schema_context(DICTIONARY, {"clean.lots", "features.lot_features"})

    assert "clean.lots" in shown
    assert "lot_features" not in shown


def test_the_domain_dictionary_describes_every_table_it_names() -> None:
    """The real one: each section is headed with a `layer.table` a view can have."""
    sections = table_sections(dictionary_path(domain_dir("coffee")).read_text(encoding="utf-8"))

    assert "clean.coffee_reviews" in sections
    assert {name.split(".")[0] for name in sections} <= {"clean", "features", "predictions"}
