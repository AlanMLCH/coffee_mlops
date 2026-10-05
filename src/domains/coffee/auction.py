"""The `auction` model's `enrich`: what a lot fetches at Mexico's Cup of Excellence auction,
against the other lots of its auction.

One item is one lot the jury ranked. What it may know is what the auction catalogue says
before the bidding: its score, whether it was a national winner (scored under the CoE's
line, sold apart), its weight, its process, its state, its varieties - Gesha above all -
and the market it was sold into: the year's mean price of other mild Arabicas.

The target is the lot's premium, in percent, over the median price of the **other** lots
of its auction, not its price. A first version priced lots over the green market instead,
and the gate refused it (2 October): from 2012 to 2026 the winners' median went from $8 to
$26 a pound, and what an auction's buyers pay as a whole moved with more than the market -
no lot's catalogue says that. What a catalogue does say is how a lot stands against the
others of its year, which is what a buyer bidding on it weighs. The level of the auction
is the other half of the price, and the API says it: the price is that median times the
premium (`relative_to`), the median of the latest auction for one not held yet.

A lot that did not sell has no target, and is scored: what it might have fetched.
"""

from collections.abc import Mapping

import polars as pl

from domains.coffee.forecast import PRICES_TABLE

LOTS_TABLE = "cup_of_excellence"
AUCTION_CONTEXT = (PRICES_TABLE, LOTS_TABLE)
MARKET = "other_milds"
CENTS = 100
# What a jury's score is rounded into, for the baseline: the premium a lot of that band
# fetched before, which is what a buyer guesses from the score alone.
SCORE_BANDS = [87.0, 88.0, 89.0, 90.0]
SCORE_LABELS = ["under 87", "87-88", "88-89", "89-90", "90 and over"]


def market_by_year(prices: pl.DataFrame) -> pl.DataFrame:
    """Each year's mean monthly price of other mild Arabicas, US dollars a pound; the
    year in course is the mean of its published months."""
    return (
        prices.filter(pl.col("frequency") == "monthly", pl.col("indicator") == MARKET)
        .group_by(pl.col("period").dt.year().cast(pl.Int64).alias("year"))
        .agg((pl.col("usd_cents_per_lb").mean() / CENTS).alias("market_usd_per_lb"))
    )


def auction_medians(items: pl.DataFrame, lots: pl.DataFrame) -> pl.DataFrame:
    """Each item with the median price of the other lots sold at its auction: never its
    own price, which is the target. A year with no auction yet takes the latest one's."""
    sold = lots.filter(pl.col("price_usd_per_lb").is_not_null()).select(
        "year", pl.col("lot_id").alias("other"), pl.col("price_usd_per_lb").alias("other_price")
    )
    others = (
        items.select("year", "lot_id")
        .join(sold, on="year", how="inner")
        .filter(pl.col("lot_id") != pl.col("other"))
        .group_by("lot_id")
        .agg(pl.col("other_price").median().alias("auction_median_usd_per_lb"))
    )
    latest = sold.filter(pl.col("year") == sold["year"].max())["other_price"].median()
    return items.join(others, on="lot_id", how="left").with_columns(
        pl.col("auction_median_usd_per_lb").fill_null(latest)
    )


def add_lot_market(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Each lot with what its catalogue says, its auction's and its market's level, and
    its premium over the other lots."""
    # A request naming no variety arrives as an empty list polars types `list[null]`, which
    # no string is "in": cast first, or every such request failed.
    varieties = (
        pl.col("varieties").cast(pl.List(pl.String)).fill_null(pl.lit([], pl.List(pl.String)))
    )
    return (
        auction_medians(items, context[LOTS_TABLE])
        .join(market_by_year(context[PRICES_TABLE]), on="year", how="left")
        .with_columns(
            pl.col("year").cast(pl.String).alias("auction"),
            pl.date(pl.col("year"), 1, 1).alias("auction_year"),
            pl.col("national_winner").cast(pl.Float64),
            varieties.list.contains("gesha").cast(pl.Float64).alias("gesha"),
            varieties.list.len().cast(pl.Float64).alias("varieties_n"),
            pl.col("score")
            .cut(SCORE_BANDS, labels=SCORE_LABELS, left_closed=True)
            .cast(pl.String)
            .alias("score_band"),
            ((pl.col("price_usd_per_lb") / pl.col("auction_median_usd_per_lb") - 1) * 100).alias(
                "premium_pct"
            ),
        )
    )
