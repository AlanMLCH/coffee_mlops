"""What an hour, or an order, of the shop may know before it happens.

An hour: its day and hour, and the menu's prices that day - the ones in force, or the ones
the owner is weighing. A request asks about an hour that has not happened, maybe at prices
the menu has never had: its `price_change_pct` moves every price from the menu in force
that day, and the model answers what that price level would bring.

An order: its slot, its items, the hands on shift and how busy the hour is. In batch, how
busy is the hour's tickets, from `shop_hours`; a request says it - the owner's expectation,
or `hourly_demand`'s - so the model can be asked about a crowd and a staffing that have not
happened.

One function each for the batch feature table and for every request to the API.
"""

from collections.abc import Mapping

import polars as pl

from domains.coffee.shops.clean import price_levels

MENU_TABLE = "menu_prices"
HOURS_TABLE = "shop_hours"
HOURS_CONTEXT = (MENU_TABLE,)
ORDERS_CONTEXT = (HOURS_TABLE,)
CHANGE = "price_change_pct"  # a request's: every price moved by it
BUSY = "tickets_in_hour"  # a request's, or the hour's own tickets


def add_hour_context(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Each hour with its weekday-and-hour slot and the price level it is asked at: the
    menu's that day against its first prices, moved by the request's change, if any."""
    change = pl.col(CHANGE) if CHANGE in items.columns else pl.lit(0.0)
    levels = price_levels(context[MENU_TABLE], items["date"].unique())
    return (
        items.drop("price_level", strict=False)
        .join(levels, on="date", how="left")
        .with_columns(
            (pl.col("price_level") * (1 + change.cast(pl.Float64) / 100)).alias("price_level"),
            pl.format(
                "{}-{}", pl.col("weekday"), pl.col("hour").cast(pl.String).str.zfill(2)
            ).alias("weekday_hour"),
        )
        .drop(CHANGE, strict=False)
    )


def add_order_context(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Each order with its weekday-and-hour slot and how busy its hour was: the tickets
    the hour brought, unless the item says how many it expects."""
    if BUSY not in items.columns:
        hours = context[HOURS_TABLE].select("date", "hour", pl.col("tickets").alias(BUSY))
        items = items.join(hours, on=["date", "hour"], how="left")
    return items.with_columns(
        pl.col(BUSY).cast(pl.Float64),
        pl.format("{}-{}", pl.col("weekday"), pl.col("hour").cast(pl.String).str.zfill(2)).alias(
            "weekday_hour"
        ),
    )
