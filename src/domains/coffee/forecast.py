"""The `green_price` model's `enrich`: a month's price change, from what was known before it.

One item is one indicator in one month (the World Bank's monthly averages: other mild
Arabicas and Robustas). The target is the month's change from the month before, in
percent - not the price itself, which the trees could not reach: 2025 and 2026 sit above
nearly every month they would learn from, and a tree predicts no higher than the
highest value it saw.

Everything an item may know is the history up to the month before it, which is what is
published when the month begins: the last price, its last three changes, the change over
twelve months, how far the last price stands from its twelve-month mean, how much it has
moved lately, and where Arabicas stand against Robustas. A forecast for September is
made with August's average, and that is the only honest use of one.

History is looked up by date, never by row: `month - 1 month`, joined. A gap in a series
then leaves a feature empty instead of quietly making it about another month.
"""

import polars as pl

PRICES_TABLE = "price_indicators"
FORECAST_INDICATORS = ("other_milds", "robustas")


def price_history(prices: pl.DataFrame) -> pl.DataFrame:
    """Per indicator and month: what was known once that month's average was out."""
    monthly = prices.filter(pl.col("frequency") == "monthly").select(
        "indicator", pl.col("period").alias("month"), pl.col("usd_cents_per_lb").alias("price")
    )
    history = (
        _back(monthly, monthly, 1, {"price": "price_1_before"})
        .pipe(_back, monthly, 12, {"price": "price_12_before"})
        .with_columns(
            change=_percent(pl.col("price"), pl.col("price_1_before")),
            change_12m=_percent(pl.col("price"), pl.col("price_12_before")),
        )
        .sort("indicator", "month")
    )
    changes = history.select("indicator", "month", "change")
    spread = monthly.pivot(on="indicator", index="month", values="price")
    ratio = (
        spread.select("month", (pl.col("other_milds") / pl.col("robustas")).alias("ratio"))
        if {"other_milds", "robustas"} <= set(spread.columns)
        else spread.select("month", pl.lit(None, pl.Float64).alias("ratio"))
    )
    return (
        history.pipe(_back, changes, 1, {"change": "change_2_back"})
        .pipe(_back, changes, 2, {"change": "change_3_back"})
        .with_columns(
            # Only over a full window: a "twelve-month mean" of the first month is that
            # month, and the distance from it a meaningless zero.
            mean_12m=pl.col("price")
            .rolling_mean_by("month", "12mo", min_samples=12)
            .over("indicator"),
            volatility_6m=pl.col("change")
            .rolling_std_by("month", "6mo", min_samples=6)
            .over("indicator"),
        )
        .join(ratio, on="month", how="left")
    )


def add_price_history(items: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Each month's item with the history of the month before it, and its change.

    Online, an item is a month to forecast: its own price is unknown, and so is the
    target. In batch, a month whose price is known but whose month before is not has no
    change to learn from, and is left out.
    """
    known = price_history(prices).select(
        "indicator",
        pl.col("month").dt.offset_by("1mo").alias("month"),  # known when this month begins
        pl.col("price").alias("price_last"),
        pl.col("change").alias("change_last"),
        "change_2_back",
        "change_3_back",
        "change_12m",
        _percent(pl.col("price"), pl.col("mean_12m")).alias("gap_to_mean_12m"),
        "volatility_6m",
        pl.col("ratio").alias("arabica_robusta_ratio"),
    )
    month = pl.col("month")
    return (
        items.filter(pl.col("frequency") == "monthly")
        .select(
            "indicator",
            pl.col("period").dt.month_start().alias("month"),
            pl.col("usd_cents_per_lb").cast(pl.Float64),
        )
        .join(known, on=["indicator", "month"], how="left")
        .with_columns(
            month_id=pl.concat_str("indicator", month.dt.strftime("%Y-%m"), separator="-"),
            decade=(month.dt.year() // 10 * 10).cast(pl.String) + "s",
            calendar_month=month.dt.strftime("%m"),
            change_pct=_percent(pl.col("usd_cents_per_lb"), pl.col("price_last")),
        )
        .filter(pl.col("usd_cents_per_lb").is_null() | pl.col("change_pct").is_not_null())
    )


def _back(
    frame: pl.DataFrame, source: pl.DataFrame, months: int, columns: dict[str, str]
) -> pl.DataFrame:
    """`frame` with `source`'s columns as they stood `months` months earlier."""
    earlier = source.select(
        "indicator",
        pl.col("month").dt.offset_by(f"{months}mo"),
        *[pl.col(old).alias(new) for old, new in columns.items()],
    )
    return frame.join(earlier, on=["indicator", "month"], how="left")


def _percent(now: pl.Expr, before: pl.Expr) -> pl.Expr:
    return (now / before - 1) * 100
