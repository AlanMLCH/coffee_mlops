"""Slicing a table without the agent: a measure, segmented by one column and coloured by
another, restricted to some values of others.

A question like "what does an item cost, area by area, for one kind of item only?" does
not need a model: it needs four choices from lists. The lists are the domain's
(`explore.datasets` in its YAML) and the query is assembled from them here, so nothing a
person types reaches the SQL but the values they pick from a column's own values -
quoted as literals. The query still runs in the agent's locked session, like everything
the explorer draws.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from decimal import Decimal

import polars as pl

from mlops_core.config import ExploreDataset

TOP = 25  # segments a bar chart shows; past that it is a list, not a comparison


def segment_sql(
    dataset: ExploreDataset,
    measure: str,
    by: str,
    color: str | None = None,
    filters: Mapping[str, Sequence[object]] | None = None,
) -> str:
    """The query for one measure of `dataset` per value of `by` (and of `color`), over the
    rows whose `filters` columns hold one of the values picked. An empty pick is no
    filter. Raises ValueError for a name the dataset does not offer."""
    if measure not in dataset.measures:
        raise ValueError(f"{dataset.name} has no measure {measure!r}")
    for role, column in (("segment", by), ("colour", color)):
        if column is not None and column not in dataset.dimensions:
            raise ValueError(f"{dataset.name} cannot be split by {column!r} ({role})")
    groups = [by] if color is None or color == by else [by, color]
    conditions = [f"({dataset.where})"] if dataset.where else []
    for column, values in (filters or {}).items():
        if column not in dataset.filters:
            raise ValueError(f"{dataset.name} cannot be filtered by {column!r}")
        if values:
            conditions.append(f"{column} IN ({', '.join(literal(v) for v in values)})")
    where = f"\nWHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        f"SELECT {', '.join(groups)}, {dataset.measures[measure]} AS {measure}\n"
        f"FROM {dataset.table}{where}\n"
        f"GROUP BY ALL\nORDER BY {', '.join(groups)}"
    )


def values_sql(dataset: ExploreDataset, column: str) -> str:
    """The values a filter can pick from: the column's own, most common first."""
    if column not in dataset.filters:
        raise ValueError(f"{dataset.name} cannot be filtered by {column!r}")
    where = f" WHERE {dataset.where}" if dataset.where else ""
    return (
        f"SELECT {column}, count(*) AS n FROM {dataset.table}{where} "
        f"GROUP BY {column} ORDER BY n DESC, {column}"
    )


def literal(value: object) -> str:
    """A value as SQL writes it; text with its quotes doubled, so it stays text."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float | Decimal):
        return repr(float(value)) if isinstance(value, Decimal) else repr(value)
    if isinstance(value, dt.datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, dt.date):
        return f"DATE '{value.isoformat()}'"
    return "'" + str(value).replace("'", "''") + "'"


def top_segments(rows: pl.DataFrame, by: str, measure: str, n: int = TOP) -> pl.DataFrame:
    """The `n` segments with the largest measure, with every colour of each: a bar chart of
    150 tasting notes says nothing a list of the first 25 does not. Time is never cut."""
    if rows.schema[by].is_temporal() or rows[by].n_unique() <= n:
        return rows
    # Ranked by each segment's largest value: a median summed over colours means nothing.
    largest = rows.group_by(by).agg(pl.col(measure).max()).sort(measure, descending=True)
    return rows.filter(pl.col(by).is_in(largest.head(n)[by].implode()))


def summary(rows: pl.DataFrame, by: str, measure: str, color: str | None = None) -> str:
    """One sentence on what the chart shows: the largest and smallest values, or the
    first and last period of a series."""
    known = rows.filter(pl.col(measure).is_not_null())
    if known.is_empty():
        return "No rows match these choices."
    label = measure.replace("_", " ")

    def named(row: dict[str, object]) -> str:
        return f"{row[by]}, {row[color]}" if color and color != by else str(row[by])

    if known.schema[by].is_temporal():
        periods = known[by].n_unique()
        first = known.filter(pl.col(by) == known[by].min()).row(0, named=True)
        last = known.filter(pl.col(by) == known[by].max()).row(0, named=True)
        return (
            f"{label}: {_number(first[measure])} in {named(first)}, "
            f"{_number(last[measure])} in {named(last)} ({periods} periods)."
        )
    ordered = known.sort(measure, descending=True)
    top, bottom = ordered.row(0, named=True), ordered.row(-1, named=True)
    if ordered.height == 1:
        return f"{label}: {_number(top[measure])} for {named(top)}."
    return (
        f"Highest {label}: {named(top)} ({_number(top[measure])}); lowest: {named(bottom)} "
        f"({_number(bottom[measure])}), across {ordered.height} rows."
    )


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"
    return f"{value:,}" if isinstance(value, int) else str(value)
