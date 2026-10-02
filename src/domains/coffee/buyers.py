"""The `households` model's `enrich`: whether a household buys coffee to drink at home.

One item is one household of INEGI's ENIGH 2024. The target is whether it paid for any
coffee - instant, beans or ground, or preparations for coffee drinks - in the week the
survey recorded its food. What it may know is what the survey knows of it besides its
coffee: its state, how many live in it, its income in the quarter and per member, and
whether it took coffee from its own harvest (a coffee farmer's household does not buy
what it grows).

A household of one sampling unit is a neighbour of the others, not an independent draw,
so the model is split by unit: no neighbourhood is on both sides. A request describes a
household; its income is asked per month, as people say it, and kept per quarter, as the
survey does.
"""

import polars as pl

HOUSEHOLDS_TABLE = "household_coffee"
PAID = ("instant_quarter_mxn", "ground_quarter_mxn", "prepared_quarter_mxn")


def add_household_traits(items: pl.DataFrame) -> pl.DataFrame:
    """Each household with what the model may know of it, and whether it bought coffee."""
    paid = pl.sum_horizontal(*[pl.col(column).fill_null(0) for column in PAID])
    members = pl.col("members").cast(pl.Float64)
    return items.with_columns(
        pl.col("year").cast(pl.String).alias("survey"),
        pl.date(pl.col("year"), 1, 1).alias("survey_year"),
        members.alias("members"),
        (pl.col("income_quarter_mxn") / members).alias("income_per_member_mxn"),
        (pl.col("own_harvest_quarter_mxn").fill_null(0) > 0).cast(pl.Float64).alias("grows_coffee"),
        pl.when(pl.all_horizontal(*[pl.col(column).is_null() for column in PAID]))
        .then(None)
        .otherwise((paid > 0).cast(pl.Float64))
        .alias("buys_coffee"),
    )
