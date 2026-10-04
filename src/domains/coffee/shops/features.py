"""What an hour of the shop may know before it happens: its day and hour, and the menu's
prices that day - the ones in force, or the ones the owner is weighing.

One function for the batch feature table and for every request to the API. A request asks
about an hour that has not happened, maybe at prices the menu has never had: its
`price_change_pct` moves every price from the menu in force that day, and the model
answers what that price level would bring.
"""

from collections.abc import Mapping

import polars as pl

from domains.coffee.shops.clean import price_levels

MENU_TABLE = "menu_prices"
HOURS_CONTEXT = (MENU_TABLE,)
CHANGE = "price_change_pct"  # a request's: every price moved by it


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
