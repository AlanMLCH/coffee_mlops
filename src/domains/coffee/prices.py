"""The international price of green coffee, day by day and month by month.

Two publishers of one family of series. The ICO computes its indicator prices daily -
the composite I-CIP and four group indicators - and publishes only the current month;
the World Bank republishes two of the group indicators (other mild Arabicas and
Robustas, ex-dock) as monthly averages back to 1960, in dollars per kilogram. One
table holds both, in the ICO's unit, each row saying how often it is observed and who
published it.

The ICO's history is every download of its page stacked: a day read twice keeps its
latest reading, since a correction can only come later. Where a month is in both, the
two are compared and the log says how far apart they are - the World Bank's month
against the mean of the ICO's days - the way the PSD file and the FAS API are.
"""

import logging
from datetime import datetime

import polars as pl

from domains.coffee.sources.ico import INDICATORS

logger = logging.getLogger(__name__)

# One US dollar per kilogram in US cents per pound: 100 cents over 2.20462 pounds.
CENTS_PER_LB_PER_USD_PER_KG = 100 * 0.45359237
# The World Bank's columns, and the ICO group indicator each one is.
WORLD_BANK_SERIES = {"Coffee, Arabica": "other_milds", "Coffee, Robusta": "robustas"}
WORLD_BANK_MONTH = "column_1"  # its months ("1960M01") head no column


def clean_price_indicators(
    daily: pl.DataFrame, monthly: pl.DataFrame, monthly_read_at: datetime
) -> pl.DataFrame:
    """One row per indicator and period: the ICO's days, then the World Bank's months."""
    days = (
        daily.unpivot(index=["date", "ingested_at"], on=list(INDICATORS), variable_name="indicator")
        .sort("ingested_at")
        .unique(["date", "indicator"], keep="last")  # a correction comes in a later read
        .select(
            pl.col("date").str.to_date().alias("period"),
            pl.lit("daily").alias("frequency"),
            "indicator",
            pl.col("value").alias("usd_cents_per_lb"),
            pl.lit("ico").alias("source"),
            pl.col("ingested_at").alias("read_at"),
        )
    )
    months = monthly.unpivot(
        index=WORLD_BANK_MONTH, on=list(WORLD_BANK_SERIES), variable_name="series"
    ).select(
        pl.col(WORLD_BANK_MONTH).str.replace("M", "-").add("-01").str.to_date().alias("period"),
        pl.lit("monthly").alias("frequency"),
        pl.col("series").replace_strict(WORLD_BANK_SERIES).alias("indicator"),
        (pl.col("value") * CENTS_PER_LB_PER_USD_PER_KG).alias("usd_cents_per_lb"),
        pl.lit("world_bank").alias("source"),
        pl.lit(monthly_read_at).alias("read_at"),
    )
    table = pl.concat([days, months]).sort("frequency", "indicator", "period")
    reconcile_prices(table)
    return table


def reconcile_prices(table: pl.DataFrame) -> dict[tuple[str, str], float]:
    """The World Bank's month minus the mean of the ICO's days, in cents per pound, for
    every indicator and month both publish; said in the log, not enforced - a month the
    ICO was read for only in part is not expected to agree."""
    daily = (
        table.filter(pl.col("frequency") == "daily")
        .group_by(pl.col("period").dt.truncate("1mo"), "indicator")
        .agg(pl.col("usd_cents_per_lb").mean().alias("days_mean"), pl.len().alias("days"))
    )
    both = table.filter(pl.col("frequency") == "monthly").join(daily, on=["period", "indicator"])
    if both.is_empty():
        logger.info("prices: no month is in both the ICO's days and the World Bank's months yet")
        return {}
    gaps = {}
    for row in both.sort("period", "indicator").iter_rows(named=True):
        gap = row["usd_cents_per_lb"] - row["days_mean"]
        gaps[(row["period"].strftime("%Y-%m"), row["indicator"])] = gap
        logger.info(
            "prices %s %s: World Bank %.2f vs the mean of %d ICO days %.2f (%+.2f cents/lb)",
            row["period"].strftime("%Y-%m"), row["indicator"], row["usd_cents_per_lb"],
            row["days"], row["days_mean"], gap,
        )  # fmt: skip
    return gaps
