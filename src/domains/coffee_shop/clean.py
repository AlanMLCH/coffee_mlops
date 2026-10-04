"""A point-of-sale export -> the coffee shop's canonical tables.

The export's text becomes typed tables - ticket lines, tickets with their timing, the
menu's price history, recipes, purchases and shifts - and one more the models read: every
hour the shop was open, with what it sold, at what price level, with how many hands. An
hour it sold nothing is a row too: a model of demand that never saw an empty hour would
never predict one.

A real shop's export arrives the same way: its columns renamed to these in its source's
`renamed` (the core does that for a file), its opening hours in the YAML.
"""

from datetime import date, datetime, timedelta

import polars as pl

from domains.coffee_shop.config import ShopConfig

_MOMENT = "%Y-%m-%dT%H:%M:%S"


def _moment(column: str) -> pl.Expr:
    return pl.col(column).str.to_datetime(_MOMENT, strict=False).cast(pl.Datetime("us"))


def clean_shop(raw: dict[str, pl.DataFrame], shop: ShopConfig) -> dict[str, pl.DataFrame]:
    """Every canonical table the export makes."""
    menu = clean_menu(raw["pos_menu"])
    shifts = clean_shifts(raw["pos_shifts"])
    sales = clean_sales(raw["pos_sales"], menu)
    orders = clean_orders(raw["pos_orders"])
    return {
        "sales": sales,
        "orders": orders,
        "menu_prices": menu,
        "recipes": clean_recipes(raw["pos_recipes"]),
        "purchases": clean_purchases(raw["pos_purchases"]),
        "shifts": shifts,
        "shop_hours": shop_hours(sales, orders, menu, shifts, shop),
    }


def clean_menu(menu: pl.DataFrame) -> pl.DataFrame:
    return menu.select(
        "product",
        "category",
        pl.col("price").cast(pl.Float64).alias("price_mxn"),
        pl.col("valid_from").str.to_date(),
        pl.col("valid_to").str.to_date(),
    ).sort("product", "valid_from")


def clean_sales(sales: pl.DataFrame, menu: pl.DataFrame) -> pl.DataFrame:
    categories = menu.select("product", "category").unique("product")
    return (
        sales.with_columns(_moment("sold_at").alias("sold_at"))
        .join(categories, on="product", how="left")
        .select(
            pl.col("ticket").alias("ticket_id"),
            pl.col("line").cast(pl.Int64),
            "sold_at",
            pl.col("sold_at").dt.date().alias("date"),
            pl.col("sold_at").dt.hour().cast(pl.Int64).alias("hour"),
            pl.col("sold_at").dt.weekday().cast(pl.Int64).alias("weekday"),
            "product",
            pl.col("category").fill_null("other"),
            pl.col("quantity").cast(pl.Int64),
            pl.col("unit_price").cast(pl.Float64).alias("unit_price_mxn"),
            (pl.col("unit_price").cast(pl.Float64) * pl.col("quantity").cast(pl.Float64)).alias(
                "line_total_mxn"
            ),
            "channel",
            "payment",
        )
        .sort("ticket_id", "line")
    )


def clean_orders(orders: pl.DataFrame) -> pl.DataFrame:
    return (
        orders.with_columns(_moment("ordered_at"), _moment("ready_at"))
        .select(
            pl.col("ticket").alias("ticket_id"),
            "ordered_at",
            "ready_at",
            pl.col("ordered_at").dt.date().alias("date"),
            pl.col("ordered_at").dt.hour().cast(pl.Int64).alias("hour"),
            ((pl.col("ready_at") - pl.col("ordered_at")).dt.total_seconds() / 60).alias("minutes"),
            pl.col("items").cast(pl.Int64),
            pl.col("baristas").cast(pl.Int64),
        )
        .sort("ticket_id")
    )


def clean_recipes(recipes: pl.DataFrame) -> pl.DataFrame:
    return recipes.select(
        "product", "ingredient", pl.col("quantity").cast(pl.Float64), "unit"
    ).sort("product", "ingredient")


def clean_purchases(purchases: pl.DataFrame) -> pl.DataFrame:
    quantity, cost = pl.col("quantity").cast(pl.Float64), pl.col("cost").cast(pl.Float64)
    return purchases.select(
        pl.col("date").str.to_date(),
        "ingredient",
        "unit",
        quantity.alias("quantity"),
        cost.alias("cost_mxn"),
        (cost / quantity).alias("unit_cost_mxn"),
    ).sort("date", "ingredient")


def clean_shifts(shifts: pl.DataFrame) -> pl.DataFrame:
    starts, ends = _moment("starts_at"), _moment("ends_at")
    hours = (ends - starts).dt.total_seconds() / 3600
    wage = pl.col("hourly_wage").cast(pl.Float64)
    return shifts.select(
        pl.col("date").str.to_date(),
        pl.col("staff").alias("staff_id"),
        "role",
        starts.alias("starts_at"),
        ends.alias("ends_at"),
        hours.alias("hours"),
        wage.alias("hourly_wage_mxn"),
        (wage * hours).alias("cost_mxn"),
    ).sort("date", "staff_id")


def shop_hours(
    sales: pl.DataFrame,
    orders: pl.DataFrame,
    menu: pl.DataFrame,
    shifts: pl.DataFrame,
    shop: ShopConfig,
) -> pl.DataFrame:
    """Every hour the shop was open, from its first sale to its last: tickets, items and
    revenue (zero when it sold nothing), the menu's price level that day, the staff on
    shift at the half hour, and how long an order took on average."""
    if sales.is_empty():
        return pl.DataFrame(schema=_HOURS)
    first, last = sales["date"].min(), sales["date"].max()
    assert isinstance(first, date) and isinstance(last, date)
    open_hours = []
    day = first
    while day <= last:
        weekday = day.isoweekday()
        if weekday in shop.opens:
            opens = datetime.combine(day, shop.opens[weekday])
            closes = datetime.combine(day, shop.closes[weekday])
            hour = opens.replace(minute=0)
            while hour < closes:
                open_hours.append({"date": day, "hour": hour.hour, "weekday": weekday})
                hour += timedelta(hours=1)
        day += timedelta(days=1)
    hours = pl.DataFrame(
        open_hours, schema={"date": pl.Date, "hour": pl.Int64, "weekday": pl.Int64}
    )
    sold = sales.group_by("date", "hour").agg(
        pl.col("ticket_id").n_unique().cast(pl.Float64).alias("tickets"),
        pl.col("quantity").sum().cast(pl.Float64).alias("items"),
        pl.col("line_total_mxn").sum().alias("revenue_mxn"),
    )
    waits = orders.group_by("date", "hour").agg(pl.col("minutes").mean().alias("mean_minutes"))
    return (
        hours.join(sold, on=["date", "hour"], how="left")
        .join(waits, on=["date", "hour"], how="left")
        .with_columns(pl.col("tickets", "items", "revenue_mxn").fill_null(0.0))
        .join(price_levels(menu, hours["date"].unique()), on="date", how="left")
        .join(staffing(shifts, hours), on=["date", "hour"], how="left")
        .with_columns(
            pl.format("{}T{}", pl.col("date"), pl.col("hour").cast(pl.String).str.zfill(2)).alias(
                "hour_id"
            ),
            pl.col("date").dt.strftime("%Y-%m").alias("month"),
            pl.col("price_level").fill_null(1.0),
            pl.col("baristas").fill_null(0),
        )
        .select(list(_HOURS))
        .sort("hour_id")
    )


def price_levels(menu: pl.DataFrame, days: pl.Series) -> pl.DataFrame:
    """Each day's menu against its first prices: the mean ratio over the products."""
    first = (
        menu.sort("valid_from").group_by("product").agg(pl.col("price_mxn").first().alias("first"))
    )
    rows = []
    for day in days.sort():
        current = menu.filter(
            pl.col("valid_from") <= day, pl.col("valid_to").is_null() | (pl.col("valid_to") >= day)
        )
        ratio = current.join(first, on="product").select(pl.col("price_mxn") / pl.col("first"))
        level = ratio.to_series().mean() if ratio.height else None
        rows.append({"date": day, "price_level": level})
    return pl.DataFrame(rows, schema={"date": pl.Date, "price_level": pl.Float64})


def staffing(shifts: pl.DataFrame, hours: pl.DataFrame) -> pl.DataFrame:
    """How many were on shift at each open hour's half hour."""
    moments = hours.with_columns(
        (
            pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=pl.col("hour"), minutes=30)
        ).alias("moment")
    )
    on = moments.join(shifts.select("date", "starts_at", "ends_at"), on="date", how="left").filter(
        (pl.col("starts_at") <= pl.col("moment")) & (pl.col("moment") < pl.col("ends_at"))
    )
    return on.group_by("date", "hour").agg(pl.len().cast(pl.Int64).alias("baristas"))


_HOURS = {
    "hour_id": pl.String,
    "date": pl.Date,
    "month": pl.String,
    "hour": pl.Int64,
    "weekday": pl.Int64,
    "tickets": pl.Float64,
    "items": pl.Float64,
    "revenue_mxn": pl.Float64,
    "price_level": pl.Float64,
    "baristas": pl.Int64,
    "mean_minutes": pl.Float64,
}
