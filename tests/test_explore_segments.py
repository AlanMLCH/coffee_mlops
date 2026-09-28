"""Slicing a table without the agent: the query is assembled from the domain's lists, and
nothing a person picks reaches it but as a quoted value."""

import datetime as dt
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from mlops_core.agent.sql import read_only, run_select
from mlops_core.config import ExploreDataset
from mlops_core.explore.segments import (
    literal,
    segment_sql,
    summary,
    top_segments,
    values_sql,
)
from mlops_core.explore.style import THEME, rgb, write_theme
from mlops_core.storage import write_table

SHELVES = ExploreDataset(
    name="shelves",
    table="clean.shelves",
    where="price > 0",
    measures={"median_price": "median(price)", "prices": "count(*)"},
    dimensions=["product", "state", "fortnight"],
    filters=["product", "state"],
)


def test_a_segment_is_one_measure_per_value_of_a_column() -> None:
    sql = segment_sql(SHELVES, "median_price", "state", "product", {"product": ["ground"]})

    assert sql == (
        "SELECT state, product, median(price) AS median_price\n"
        "FROM clean.shelves\n"
        "WHERE (price > 0) AND product IN ('ground')\n"
        "GROUP BY ALL\nORDER BY state, product"
    )


def test_an_empty_pick_is_no_filter_and_the_same_column_twice_is_once() -> None:
    dataset = SHELVES.model_copy(update={"where": None})

    sql = segment_sql(dataset, "prices", "state", "state", {"product": []})

    assert (
        sql == "SELECT state, count(*) AS prices\nFROM clean.shelves\nGROUP BY ALL\nORDER BY state"
    )


@pytest.mark.parametrize(
    ("measure", "by", "color", "filters", "problem"),
    [
        ("mean_price", "state", None, {}, "no measure 'mean_price'"),
        ("prices", "store", None, {}, "cannot be split by 'store'"),
        ("prices", "state", "store", {}, "cannot be split by 'store' (colour)"),
        ("prices", "state", None, {"fortnight": ["x"]}, "cannot be filtered by 'fortnight'"),
    ],
)
def test_only_the_dataset_s_own_names_are_accepted(
    measure: str, by: str, color: str | None, filters: dict[str, list[object]], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem.replace("(", r"\(").replace(")", r"\)")):
        segment_sql(SHELVES, measure, by, color, filters)
    with pytest.raises(ValueError, match="cannot be filtered by 'fortnight'"):
        values_sql(SHELVES, "fortnight")


def test_a_picked_value_stays_a_value() -> None:
    assert literal("Liverpool'); DROP TABLE x; --") == "'Liverpool''); DROP TABLE x; --'"
    assert [literal(v) for v in (None, True, False, 3, 2.5, Decimal("1.5"))] == [
        "NULL", "TRUE", "FALSE", "3", "2.5", "1.5"
    ]  # fmt: skip
    assert literal(dt.date(2026, 7, 1)) == "DATE '2026-07-01'"
    assert literal(dt.datetime(2026, 7, 1, 8, 30)) == "TIMESTAMP '2026-07-01 08:30:00'"


def test_a_dataset_names_bare_columns_only() -> None:
    with pytest.raises(ValidationError, match="not bare column names"):
        ExploreDataset.model_validate(
            SHELVES.model_dump() | {"dimensions": ["state; DROP TABLE x"]}
        )
    with pytest.raises(ValidationError, match="table"):
        ExploreDataset.model_validate(SHELVES.model_dump() | {"table": "shelves"})


def test_the_query_runs_in_the_locked_session(tmp_path: Path) -> None:
    shelves = pl.DataFrame(
        {
            "product": ["ground", "ground", "instant", "ground"],
            "state": ["Chiapas", "Oaxaca", "Chiapas", "Oaxaca"],
            "fortnight": [dt.date(2026, 7, 1)] * 4,
            "price": [380.0, 400.0, 900.0, -1.0],
        }
    )
    write_table(shelves, tmp_path / "clean" / "shelves", {})
    con = read_only(tmp_path)

    rows = run_select(
        con, segment_sql(SHELVES, "median_price", "state", None, {"product": ["ground"]})
    )
    values = run_select(con, values_sql(SHELVES, "product"))

    assert rows.rows == [("Chiapas", 380.0), ("Oaxaca", 400.0)]  # the -1 is not a price
    assert values.rows == [("ground", 2), ("instant", 1)]


def test_the_largest_segments_are_kept_and_time_never_cut() -> None:
    notes = pl.DataFrame({"note": [f"n{i}" for i in range(30)], "coffees": list(range(30))})
    series = pl.DataFrame(
        {"day": [dt.date(2026, 1, d) for d in range(1, 31)], "coffees": list(range(30))}
    )

    top = top_segments(notes, "note", "coffees", n=3)

    assert top["note"].to_list() == ["n27", "n28", "n29"]
    assert top_segments(series, "day", "coffees", n=3).height == 30
    assert top_segments(notes.head(2), "note", "coffees", n=3).height == 2


def test_the_summary_says_what_the_chart_shows() -> None:
    by_state = pl.DataFrame(
        {"state": ["Chiapas", "Oaxaca", "Puebla"], "median_price": [380.0, 12.5, None]}
    )
    coloured = by_state.with_columns(product=pl.lit("ground"))
    series = pl.DataFrame(
        {"fortnight": [dt.date(2026, 1, 1), dt.date(2026, 7, 16)], "prices": [3, 5]}
    )

    assert summary(by_state, "state", "median_price") == (
        "Highest median price: Chiapas (380); lowest: Oaxaca (12.50), across 2 rows."
    )
    assert summary(coloured, "state", "median_price", "product").startswith(
        "Highest median price: Chiapas, ground (380)"
    )
    assert summary(series, "fortnight", "prices") == (
        "prices: 3 in 2026-01-01, 5 in 2026-07-16 (2 periods)."
    )
    assert summary(by_state.head(1), "state", "median_price") == "median price: 380 for Chiapas."
    assert summary(by_state.tail(1), "state", "median_price") == "No rows match these choices."


def test_the_theme_is_written_as_a_file_streamlit_reads(tmp_path: Path) -> None:
    import tomllib

    written = tomllib.loads(write_theme(tmp_path / "theme.toml").read_text(encoding="utf-8"))

    assert written == {"theme": THEME}  # lists stay lists, which the command line cannot do
    assert rgb("#B5653A") == (181, 101, 58)
