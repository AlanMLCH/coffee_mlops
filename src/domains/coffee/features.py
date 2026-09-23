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
ORIGIN_ATTRIBUTES = ["country", "state", "processing_method", "producer"]
# Varieties a coffee may be given a column of its own for. One summarised variety is not
# enough: a coffee whose sheet lists three says "multiple" (172 of the 510 offers do),
# and a Gesha in a blend then looks like any other blend. Chosen by how often each
# appears in the catalogues - every variety in at least four coffees, read 2026-09-23 -
# and never by what it sells for, which is the target.
VARIETY_FEATURES = [
    "bourbon",
    "caturra",
    "colombia",
    "garnica",
    "gesha",
    "heirloom",
    "jember",
    "marsellesa",
    "mundo novo",
    "oro azteca",
    "pluma mejorado",
    "ruiru 11",
    "sarchimor",
    "sl28",
    "sl34",
    "typica",
]


def variety_column(variety: str) -> str:
    """`gesha` -> `variety_gesha`, the column name the model's config declares."""
    return f"variety_{variety.replace(' ', '_')}"


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
    (across every variety any origin lists), a column per notable variety, how many
    origins and varieties it names, and the altitude as the mean of their ranges'
    midpoints."""
    midpoint = (pl.col("altitude_min_m") + pl.col("altitude_max_m")) / 2
    attributes = origins.group_by("coffee_id").agg(
        *[_agreed(column) for column in ORIGIN_ATTRIBUTES],
        midpoint.mean().alias("altitude_m"),
        pl.len().alias("origins_n"),  # more than one is a blend
    )
    listed = (
        origins.select("coffee_id", "varieties")
        .explode("varieties")
        .rename({"varieties": "variety"})
    )
    # A sheet that lists no variety knows nothing about varieties: its counters and flags
    # are null, not zero. Zero would say "this coffee is not a Gesha", which is a claim
    # the sheet never made.
    stated = pl.col("variety").drop_nulls()
    names_one = stated.len() > 0
    varieties = listed.group_by("coffee_id").agg(
        _agreed("variety"),
        pl.when(names_one).then(stated.n_unique()).cast(pl.Float64).alias("varieties_n"),
        *[
            pl.when(names_one)
            .then(pl.col("variety").eq(variety).any())
            .cast(pl.Float64)
            .alias(variety_column(variety))
            for variety in VARIETY_FEATURES
        ],
    )
    return attributes.join(varieties, on="coffee_id", how="left")


def _as_stated() -> list[pl.Expr]:
    """The summary columns for a request that describes one coffee: it has one origin,
    and the variety it names is the only one it has - or none, which is not "not a Gesha"."""
    named = pl.col("variety").is_not_null()
    return [
        pl.lit(1.0).alias("origins_n"),
        pl.when(named).then(1.0).alias("varieties_n"),
        *[
            pl.when(named)
            .then(pl.col("variety").eq(variety))
            .cast(pl.Float64)
            .alias(variety_column(variety))
            for variety in VARIETY_FEATURES
        ],
    ]


def add_coffee_origin(items: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Offers take their coffee's origin; an online request states its own.

    A request describes a coffee no catalogue has to list, so it carries its attributes
    and no `coffee_id`; nothing is looked up for it, but the columns a summary would have
    produced are derived from what it states, or online and batch would not agree on what
    the model is fed. Offers read from the catalogues are examples only with a price to
    learn from: none without a size, and none whose price was flagged as copied from
    another size.
    """
    if "coffee_id" not in items.columns:
        return items.with_columns(_as_stated())
    priced = items.filter(
        pl.col("price_mxn_per_kg").is_not_null() & ~pl.col("price_outlier").fill_null(False)
    )
    return priced.join(coffee_origins(context), on="coffee_id", how="left")
