"""What a lot may know about its origin's market, and when it may know it.

A lot graded in year Y sees market year Y-1: the latest balance that was complete when
it was cupped. Seeing Y itself would be leakage - that year's numbers were not published
yet - and a model trained with it would look better than it could ever be in service.

This is the domain's `enrich`, and the only place the join exists: the batch feature
table and every online request go through the same function, so the two cannot compute
a feature differently. A test holds them to it.
"""

import polars as pl

CONTEXT_TABLE = "market_context"


def market_features(context: pl.DataFrame) -> pl.DataFrame:
    """One row per (country, market_year) with the context features."""
    production = pl.col("production")
    return context.select(
        "country",
        "market_year",
        production.alias("ctx_production"),
        pl.when(production > 0)
        .then(pl.col("arabica_production") / production)
        .alias("ctx_arabica_share"),
        # Can exceed 1: re-exports and stock drawdowns.
        pl.when(production > 0).then(pl.col("exports") / production).alias("ctx_export_share"),
        pl.col("domestic_consumption").alias("ctx_domestic_consumption"),
    )


def add_market_context(items: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Point-in-time join: an item graded in year Y sees market year Y-1."""
    market_year = (pl.col("grading_date").dt.year() - 1).alias("market_year")
    return items.with_columns(market_year).join(
        market_features(context), on=["country", "market_year"], how="left"
    )
