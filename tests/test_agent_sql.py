"""The agent's SQL session and the schema it is shown.

Every refusal below was first seen to succeed without its guard: DuckDB with external
access off still let `COPY ... TO` write inside an allowed directory, which is why the
parser check exists at all.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import polars as pl
import pytest

from mlops_core import cli
from mlops_core.adapter import domain_dir, load_adapter
from mlops_core.agent.dictionary import (
    SchemaLinker,
    dictionary_path,
    schema_context,
    table_sections,
)
from mlops_core.agent.sql import Refused, qualified_views, read_only, run_select, views
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


def test_queries_from_many_threads_run_at_once_and_keep_the_guards(
    session: duckdb.DuckDBPyConnection,
) -> None:
    """Every query takes a cursor of its own: one that runs too long is interrupted alone,
    the others answer, and a cursor is as locked as the session it came from."""
    slow = "SELECT count(*) FROM range(1000000000000) a"
    tasks = [(slow, 0.3)] + [("SELECT count(*) AS n FROM clean.lots", 5.0)] * 6

    def run(task: tuple[str, float]) -> object:
        try:
            return run_select(session, task[0], timeout=task[1]).rows
        except duckdb.InterruptException:
            return "interrupted"

    with ThreadPoolExecutor(len(tasks)) as pool:
        answers = list(pool.map(run, tasks))

    assert answers[0] == "interrupted" and len({str(a) for a in answers[1:]}) == 1
    with pytest.raises(duckdb.PermissionException):
        run_select(session, "SELECT * FROM read_csv('https://example.com/x.csv')")


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
    # Nor a parent's, lent to a subdomain under the parent's name.
    lent = DICTIONARY.replace("`features.lot_features`", "`market.features.lot_features`")
    assert "lot_features" not in schema_context(lent, {"market.features.lot_features"})


def test_the_domain_dictionary_describes_every_table_it_names() -> None:
    """The real one: each section is headed with a `layer.table` a view can have."""
    sections = table_sections(dictionary_path(domain_dir("coffee")).read_text(encoding="utf-8"))

    assert "clean.coffee_reviews" in sections
    assert {name.split(".")[0] for name in sections} <= {
        "clean",
        "features",
        "predictions",
        "analysis",  # the studies an owner asks about: unseen, the agent could answer none
    }


def test_a_question_is_shown_the_sections_like_it_and_the_ones_they_name() -> None:
    """Embedded as one-hot words: the question about prices meets the prices' section,
    which names the rates table it joins - so the rates come too, and the lots do not."""
    sections = {
        "clean.lots": "## `clean.lots` - one lot",
        "clean.prices": "## `clean.prices` - one price; join `clean.rates` for pesos",
        "clean.rates": "## `clean.rates` - one rate",
    }
    words = ["lot", "price", "rate"]

    def embed(texts: list[str]) -> np.ndarray:
        return np.array([[float(w in text.lower()) for w in words] for text in texts])

    linker = SchemaLinker(sections, embed, k=1, query=lambda q: q.lower())

    assert linker.chosen("What does a price cost?") == ["clean.prices", "clean.rates"]
    assert linker("What does a price cost?") == "\n\n".join(
        [sections["clean.prices"], sections["clean.rates"]]
    )


def test_the_sql_writer_is_linked_only_when_the_domain_asks() -> None:
    config = load_adapter("coffee").config
    dictionary = "## `clean.lots` - one lot\n\n## `clean.prices` - one price"
    names = {"clean.lots", "clean.prices"}
    embedder = SimpleNamespace(embed=lambda texts: np.ones((len(texts), 2)))
    linked = config.model_copy(
        update={"agent": config.agent.model_copy(update={"schema_sections": 1})}
    )

    assert config.agent.schema_sections is None
    assert cli._linker(config, dictionary, names, embedder) is None
    assert isinstance(cli._linker(linked, dictionary, names, embedder), SchemaLinker)


def test_a_parents_table_written_without_its_domain_is_qualified() -> None:
    """A subdomain's small model drops the parent's name; when the table can only be one
    of the parent's, it is named as the session has it - and nothing else is touched."""
    names = {"clean.sales", "market.clean.places", "market.analysis.trend", "other.analysis.trend"}

    assert qualified_views("SELECT count(*) FROM clean.places p", names) == (
        "SELECT count(*) FROM market.clean.places p"
    )
    assert qualified_views("SELECT * FROM clean.sales", names) == "SELECT * FROM clean.sales"
    # Two lent tables could be meant: left as written, for the error to say so.
    assert qualified_views("SELECT * FROM analysis.trend", names) == "SELECT * FROM analysis.trend"
    assert qualified_views("SELECT s.clean FROM clean.unknown s", names) == (
        "SELECT s.clean FROM clean.unknown s"
    )


def test_a_table_put_in_another_layer_or_domain_is_named_as_the_session_has_it() -> None:
    """The table's name is right and its layer, or its domain, is not: when only one view
    has that table, it is the one meant."""
    names = {"clean.sales", "analysis.alerts", "market.clean.places", "market.analysis.trend"}

    assert qualified_views("SELECT * FROM clean.alerts a", names) == (
        "SELECT * FROM analysis.alerts a"
    )
    assert qualified_views("SELECT * FROM market.clean.alerts", names) == (
        "SELECT * FROM analysis.alerts"
    )
    assert qualified_views("SELECT * FROM market.analysis.places JOIN clean.trend", names) == (
        "SELECT * FROM market.clean.places JOIN market.analysis.trend"
    )
    # An alias named like a layer: its columns are no table's name, and are left alone.
    assert qualified_views("SELECT clean.quantity FROM clean.sales clean", names) == (
        "SELECT clean.quantity FROM clean.sales clean"
    )
