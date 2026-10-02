"""The `green_range` model's `enrich`: where an international green coffee price could be
3, 6 or 12 months on, from what is known now.

One item is one indicator, one month and one horizon: the change, in percent, from the
price `h` months before the month to the month's own. What it may know is what was known
once that earlier month's average was out - the same history `green_price` reads
(`domains.coffee.forecast.price_history`, looked up by date) - and how far ahead it is
asked. With `h` = 1 it would be `green_price`'s item; this model asks further ahead, and
for a range instead of a point.

Beyond the last published month, the months 3, 6 and 12 ahead are items too, with no
price yet: the batch scores them, and that is the outlook the explorer draws. A request
names an indicator and a horizon, and is exactly that item.
"""

from collections.abc import Mapping

import polars as pl

from domains.coffee.forecast import PRICES_TABLE, price_history

HORIZONS = (3, 6, 12)


def known_at(prices: pl.DataFrame) -> pl.DataFrame:
    """Per indicator and month: what was known once that month's average was out."""
    history = price_history(prices)
    return history.select(
        "indicator",
        pl.col("month").alias("origin"),
        pl.col("price").alias("price_last"),
        pl.col("change").alias("change_last"),
        "change_2_back",
        "change_3_back",
        "change_12m",
        ((pl.col("price") / pl.col("mean_12m") - 1) * 100).alias("gap_to_mean_12m"),
        "volatility_6m",
        pl.col("ratio").alias("arabica_robusta_ratio"),
    )


def add_price_outlook(items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Each month at each horizon, and each horizon ahead of the last published month.

    A month whose price is known has its origin `h` months before it; an item without one
    - a request, or a month ahead - starts from the indicator's last published month."""
    prices = context[PRICES_TABLE]
    monthly = items.filter(pl.col("frequency") == "monthly").select(
        "indicator",
        pl.col("period").cast(pl.Date).dt.month_start().alias("month"),  # a request has none
        pl.col("usd_cents_per_lb").cast(pl.Float64),
        *(["horizon_months"] if "horizon_months" in items.columns else []),
    )
    if "horizon_months" not in monthly.columns:  # the batch: every month, and the months ahead
        horizons = pl.DataFrame({"horizon_months": HORIZONS}, schema={"horizon_months": pl.Int64})
        ahead = (
            monthly.select("indicator")
            .unique()
            .join(horizons, how="cross")
            .with_columns(
                pl.lit(None, pl.Date).alias("month"),
                pl.lit(None, pl.Float64).alias("usd_cents_per_lb"),
            )
        )
        monthly = pl.concat(
            [
                monthly.join(horizons, how="cross"),
                ahead.select(*monthly.columns, "horizon_months"),
            ]
        )
    latest = (
        prices.filter(pl.col("frequency") == "monthly", pl.col("usd_cents_per_lb").is_not_null())
        .group_by("indicator")
        .agg(pl.col("period").max().dt.month_start().alias("latest"))
    )
    h = pl.col("horizon_months").cast(pl.Int64)
    month = pl.col("month")
    return (
        monthly.with_columns(h)
        .join(latest, on="indicator", how="left")
        .with_columns(
            pl.when(month.is_null())
            .then(pl.col("latest"))
            .otherwise(month.dt.offset_by(pl.format("-{}mo", h)))
            .alias("origin")
        )
        .with_columns(
            pl.when(month.is_null())
            .then(pl.col("origin").dt.offset_by(pl.format("{}mo", h)))
            .otherwise(month)
            .alias("month")
        )
        .join(known_at(prices), on=["indicator", "origin"], how="left")
        .with_columns(
            pl.concat_str(
                "indicator", month.dt.strftime("%Y-%m"), pl.format("{}m", h), separator="-"
            ).alias("outlook_id"),
            month.dt.year().cast(pl.String).alias("year"),
            month.dt.strftime("%m").alias("calendar_month"),
            ((pl.col("usd_cents_per_lb") / pl.col("price_last") - 1) * 100).alias("change_pct"),
            h.cast(pl.Float64).alias("horizon_months"),
        )
        .filter(pl.col("usd_cents_per_lb").is_null() | pl.col("change_pct").is_not_null())
    )


OUTLOOK_COVERAGE = 0.8


def price_outlook(prices: pl.DataFrame) -> pl.DataFrame:
    """Where each price could be 3, 6 and 12 months after its last published month: the
    range its own changes over that horizon have spanned, four times in five, since the
    series began, set on the last price.

    It is the baseline the `green_range` model was held against, and the one it could not
    beat (2 October): the model's ranges were narrower and held the 2021-2026 changes 71%
    of the time, not 80%. A market's own history gives a range; it does not say where in
    it the next year falls."""
    tail = (1 - OUTLOOK_COVERAGE) / 2
    monthly = prices.filter(
        pl.col("frequency") == "monthly", pl.col("usd_cents_per_lb").is_not_null()
    ).select(
        "indicator",
        pl.col("period").cast(pl.Date).dt.month_start().alias("month"),
        pl.col("usd_cents_per_lb").alias("price"),
    )
    rows = []
    for (indicator,), series in monthly.group_by("indicator", maintain_order=True):
        last = series.sort("month").row(-1, named=True)
        for h in HORIZONS:
            before = series.select(
                pl.col("month").dt.offset_by(f"{h}mo"), pl.col("price").alias("before")
            )
            changes = (
                series.join(before, on="month", how="inner")
                .select((pl.col("price") / pl.col("before") - 1) * 100)
                .to_series()
            )
            if changes.is_empty():  # a series shorter than the horizon says nothing of it
                continue
            low, mid, high = (float(changes.quantile(q)) for q in (tail, 0.5, 1 - tail))  # type: ignore[arg-type]
            rows.append(
                {"indicator": indicator, "horizon_months": h, "from_month": last["month"],
                 "to_month": last["month"].replace(day=1), "price_now": last["price"],
                 "change_low_pct": low, "change_median_pct": mid, "change_high_pct": high,
                 "months": changes.len()}
            )  # fmt: skip
    if not rows:
        return pl.DataFrame(schema=OUTLOOK)
    return (
        pl.DataFrame(rows, schema=OUTLOOK)
        .with_columns(pl.col("to_month").dt.offset_by(pl.format("{}mo", "horizon_months")))
        .with_columns(
            *[
                (pl.col("price_now") * (1 + pl.col(f"change_{edge}_pct") / 100)).alias(
                    f"price_{edge}"
                )
                for edge in ("low", "median", "high")
            ]
        )
    )


OUTLOOK = {
    "indicator": pl.String,
    "horizon_months": pl.Int64,
    "from_month": pl.Date,
    "to_month": pl.Date,
    "price_now": pl.Float64,
    "change_low_pct": pl.Float64,
    "change_median_pct": pl.Float64,
    "change_high_pct": pl.Float64,
    "months": pl.Int64,
}
