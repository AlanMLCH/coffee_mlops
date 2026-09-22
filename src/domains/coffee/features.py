"""The domain's `enrich`, one per model: what each item may know, and when.

- `review`: a lot graded in year Y sees its origin's market year Y-1, the latest balance
  that was complete when it was cupped. Seeing Y itself would be leakage - that year's
  numbers were not published yet - and a model trained with it would look better than it
  could ever be in service.
- `offer`: a bag on a shop's shelf is described by its coffee's sheet. Its origins are
  summarised to one value per attribute; a blend whose origins disagree says "multiple".

These are the only places the joins exist: the batch feature table and every online
request go through the same function, so the two cannot compute a feature differently.
A test holds them to it.
"""

import polars as pl

CONTEXT_TABLE = "market_context"
ORIGINS_TABLE = "roaster_origins"
# A blend whose origins disagree on an attribute: not unknown, and not any one of them.
MULTIPLE = "multiple"
# What an offer's coffee says about where it grew, summarised to one value per coffee.
ORIGIN_ATTRIBUTES = ["country", "state", "processing_method"]


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


def _agreed(column: str) -> pl.Expr:
    """The value a coffee's origins agree on; "multiple" if they differ; null if none says."""
    stated = pl.col(column).drop_nulls()
    return (
        pl.when(stated.n_unique() == 1)
        .then(stated.first())
        .when(stated.n_unique() > 1)
        .then(pl.lit(MULTIPLE))
        .alias(column)
    )


def coffee_origins(origins: pl.DataFrame) -> pl.DataFrame:
    """One row per coffee: each attribute its origins agree on, the variety the same way
    (across every variety any origin lists), and the altitude as the mean of their
    ranges' midpoints."""
    midpoint = (pl.col("altitude_min_m") + pl.col("altitude_max_m")) / 2
    attributes = origins.group_by("coffee_id").agg(
        *[_agreed(column) for column in ORIGIN_ATTRIBUTES], midpoint.mean().alias("altitude_m")
    )
    varieties = (
        origins.select("coffee_id", "varieties")
        .explode("varieties")
        .rename({"varieties": "variety"})
        .group_by("coffee_id")
        .agg(_agreed("variety"))
    )
    return attributes.join(varieties, on="coffee_id", how="left")


def add_coffee_origin(items: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Offers take their coffee's origin; an online request states its own.

    A request describes a coffee no catalogue has to list, so it carries its attributes
    and no `coffee_id`; nothing is looked up for it. Offers read from the catalogues are
    examples only with a price to learn from: none without a size, and none whose price
    was flagged as copied from another size.
    """
    if "coffee_id" not in items.columns:
        return items
    priced = items.filter(
        pl.col("price_mxn_per_kg").is_not_null() & ~pl.col("price_outlier").fill_null(False)
    )
    return priced.join(coffee_origins(context), on="coffee_id", how="left")
