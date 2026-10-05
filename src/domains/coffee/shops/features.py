"""What an hour, or an order, of the shop may know before it happens.

An hour: its day and hour, the menu's prices that day - the ones in force, or the ones the
owner is weighing - and how the same weekday-and-hour went the last times it came round,
at what prices. A request asks about an hour that has not happened, maybe at prices the
menu has never had: its `price_change_pct` moves every price from the menu in force that
day, and the model answers what that price level would bring. Its history is the last
weeks the shop's tables hold before its day: last week's, for next week's hour.

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
HOURS_CONTEXT = (MENU_TABLE, HOURS_TABLE)
HISTORY_WEEKS = 4  # the same slot's last times a forecast reads
ORDERS_CONTEXT = (HOURS_TABLE,)
CHANGE = "price_change_pct"  # a request's: every price moved by it
BUSY = "tickets_in_hour"  # a request's, or the hour's own tickets


def add_hour_context(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Each hour with its weekday-and-hour slot, the price level it is asked at - the
    menu's that day against its first prices, moved by the request's change, if any - and
    its slot's history (`slot_history`)."""
    change = pl.col(CHANGE) if CHANGE in items.columns else pl.lit(0.0)
    levels = price_levels(context[MENU_TABLE], items["date"].unique())
    return (
        with_history(items.drop("price_level", strict=False), context[HOURS_TABLE])
        .join(levels, on="date", how="left")
        .with_columns(
            (pl.col("price_level") * (1 + change.cast(pl.Float64) / 100)).alias("price_level"),
            pl.format(
                "{}-{}", pl.col("weekday"), pl.col("hour").cast(pl.String).str.zfill(2)
            ).alias("weekday_hour"),
        )
        .drop(CHANGE, strict=False)
    )


def slot_history(hours: pl.DataFrame) -> pl.DataFrame:
    """For each open hour, its slot's history up to and including it: its own tickets
    (`slot_last`), the mean of its last `HISTORY_WEEKS` (`slot_mean_4w`) and the menu's
    level over them (`history_price_level`)."""
    slot = ("weekday", "hour")

    def mean_of_last(column: str) -> pl.Expr:
        return pl.col(column).rolling_mean(HISTORY_WEEKS, min_samples=1).over(slot)

    return (
        hours.select("date", *slot, "tickets", "price_level")
        .sort("date")
        .with_columns(
            pl.col("tickets").alias("slot_last"),
            mean_of_last("tickets").alias("slot_mean_4w"),
            mean_of_last("price_level").alias("history_price_level"),
        )
        .select("date", *slot, "slot_last", "slot_mean_4w", "history_price_level")
    )


def with_history(items: pl.DataFrame, hours: pl.DataFrame) -> pl.DataFrame:
    """Each item with its slot's history as it stood before its day: the latest open hour
    of the same weekday and hour strictly earlier - last week's, in the tables; the latest
    the tables hold, for a day past them. Never its own day: that is the target."""
    order = "_row"
    ordered = items.with_row_index(order).sort("date")
    joined = ordered.join_asof(
        slot_history(hours),
        on="date",
        by=["weekday", "hour"],
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,  # sorted just above; polars cannot check it within groups
    )
    return joined.sort(order).drop(order)


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
