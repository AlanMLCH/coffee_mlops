"""What a shop's own data says, for its owner: what each product earns and how inflation and
green coffee move it, how its customers answer a price and what a change would do, when to
raise, which products carry the menu, where its hands fall short, and how its corner of the
city compares with the rest.

Studies, not models: each is a small, explainable computation an owner can check by hand,
with its uncertainty where it has one. A price change is weighed with an elasticity fitted
on the shop's own price history - a structural assumption said as such - because a tree
fitted on that history cannot say what a price it never saw would do.
"""

import math
from collections.abc import Collection, Mapping
from datetime import date

import numpy as np
import polars as pl

from domains.coffee.shops.config import CoffeeShopConfig, ShopConfig, StudiesConfig

# Menu engineering's classic line (Kasavana and Smith, 1982): a product is popular when it
# sells at least 70% of an equal share of the items.
POPULAR_SHARE = 0.7
EARTH_RADIUS_M = 6_371_000.0
COFFEE = "coffee"  # the kind of place the parent's register calls a coffee shop
OWN_TABLES = ("sales", "orders", "recipes", "purchases", "shifts", "shop_hours")

PRICE_RESPONSE = {
    "scope": pl.String,  # "whole day", or "quiet hours" (slots whose queue stays short)
    "elasticity": pl.Float64,  # % change in tickets for a 1% change in every price
    "elasticity_low": pl.Float64,
    "elasticity_high": pl.Float64,
    "days": pl.Int64,
    "price_levels": pl.Int64,  # distinct menu levels the history holds
    "level_min": pl.Float64,
    "level_max": pl.Float64,
}
SCENARIOS = {
    "change_pct": pl.Float64,
    "price_level": pl.Float64,
    "within_history": pl.Boolean,  # a level the menu has had: no extrapolation
    "tickets_per_day": pl.Float64,
    "revenue_per_day": pl.Float64,
    "gross_profit_per_day": pl.Float64,  # revenue less what went into what was sold
    "gross_profit_low": pl.Float64,
    "gross_profit_high": pl.Float64,
    "versus_now_pct": pl.Float64,
    "versus_now_low_pct": pl.Float64,
    "versus_now_high_pct": pl.Float64,
}
NEIGHBOURHOOD = {
    "measure": pl.String,
    "shop": pl.Float64,  # the shop's, or its AGEB's
    "borough": pl.Float64,  # the median AGEB of its borough
    "city": pl.Float64,  # the median AGEB of the city
    "unit": pl.String,
}


def studies(clean: Mapping[str, pl.DataFrame], config: CoffeeShopConfig) -> dict[str, pl.DataFrame]:
    """Every study of one shop, from its clean tables and the parent's it lists; none
    before its tables are built."""
    if any(table not in clean for table in OWN_TABLES):
        return {}
    shop, settings = config.shop, config.studies
    beans = beans_of(shop)
    margins = product_margins(clean["sales"], clean["recipes"], clean["purchases"], beans)
    response = price_response(
        clean["shop_hours"],
        clean["orders"],
        shop.wait_target_minutes,
        settings.resamples,
        settings.seed,
    )
    cpi = clean.get(config.simulation.inflation)
    return {
        "product_margins": margins,
        "price_response": response,
        "price_scenarios": price_scenarios(
            clean["shop_hours"], clean["sales"], margins, response, settings
        ),
        "price_alerts": price_alerts(
            margins,
            shop,
            clean.get(settings.green_outlook),
            cpi,
            config.simulation.green_indicator,
            settings.outlook_months,
        ),
        "inflation_impact": inflation_impact(clean["shop_hours"], margins, clean["shifts"], cpi),
        "menu_engineering": menu_engineering(margins, settings.menu_months),
        "staffing": staffing(
            clean["shop_hours"], clean["orders"], clean["shifts"], shop.wait_target_minutes
        ),
        "neighbourhood": neighbourhood(shop, clean, clean["purchases"], beans, settings),
    }


def beans_of(shop: ShopConfig) -> set[str]:
    """The ingredients whose cost follows green coffee: the shop's coffee."""
    return {name for name, item in shop.ingredients.items() if item.follows == "green_coffee"}


# --- What each product earns -------------------------------------------------------------


def product_margins(
    sales: pl.DataFrame,
    recipes: pl.DataFrame,
    purchases: pl.DataFrame,
    beans: Collection[str],
) -> pl.DataFrame:
    """Each product, each month: the price it sold at, what went into one, and what was left.

    An ingredient costs what the month's purchases paid a unit (the last month's, when it
    bought none), times what the shop buys for each unit it uses - its waste, read from the
    whole history: everything bought against what the recipes say was sold. What is thrown
    away is part of what a product costs. `beans` are the ingredients whose share of the
    cost is reported: the coffee in it."""
    month = pl.col("date").dt.truncate("1mo").alias("month")
    sold = sales.group_by(month, "product", "category").agg(
        pl.col("quantity").sum().alias("units"),
        (pl.col("line_total_mxn").sum() / pl.col("quantity").sum()).alias("price_mxn"),
    )
    used = (
        sales.group_by("product")
        .agg(pl.col("quantity").sum().alias("units"))
        .join(recipes, on="product")
        .group_by("ingredient")
        .agg((pl.col("units") * pl.col("quantity")).sum().alias("used"))
    )
    waste = (
        purchases.group_by("ingredient")
        .agg(pl.col("quantity").sum().alias("bought"))
        .join(used, on="ingredient")
        .select("ingredient", (pl.col("bought") / pl.col("used")).alias("bought_per_used"))
    )
    months = sold.select("month").unique()
    unit_costs = purchases.group_by(month, "ingredient").agg(
        (pl.col("cost_mxn").sum() / pl.col("quantity").sum()).alias("unit_cost")
    )
    filled = (
        months.join(unit_costs.select("ingredient").unique(), how="cross")
        .join(unit_costs, on=["month", "ingredient"], how="full", coalesce=True)
        .sort("ingredient", "month")
        .with_columns(pl.col("unit_cost").forward_fill().backward_fill().over("ingredient"))
        .join(waste, on="ingredient", how="left")
        .with_columns(
            (pl.col("unit_cost") * pl.col("bought_per_used").fill_null(1.0)).alias("used_cost")
        )
    )
    costs = (
        sold.select("month", "product")
        .join(recipes, on="product")
        .join(filled.select("month", "ingredient", "used_cost"), on=["month", "ingredient"])
        .with_columns((pl.col("quantity") * pl.col("used_cost")).alias("cost"))
        .group_by("month", "product")
        .agg(
            pl.col("cost").sum().alias("unit_cost_mxn"),
            pl.col("cost").filter(pl.col("ingredient").is_in(list(beans))).sum().alias("beans"),
        )
    )
    return (
        sold.join(costs, on=["month", "product"], how="left")
        .with_columns(
            (pl.col("price_mxn") - pl.col("unit_cost_mxn")).alias("margin_mxn"),
            (100 * (1 - pl.col("unit_cost_mxn") / pl.col("price_mxn"))).alias("margin_pct"),
            (100 * pl.col("beans") / pl.col("unit_cost_mxn")).alias("beans_share_pct"),
        )
        .select(
            "month",
            "product",
            "category",
            "units",
            "price_mxn",
            "unit_cost_mxn",
            "margin_mxn",
            "margin_pct",
            "beans_share_pct",
        )
        .sort("month", "product")
    )


# --- How customers answer a price, and what a change would do -----------------------------


def price_response(
    hours: pl.DataFrame,
    orders: pl.DataFrame,
    wait_target: float,
    resamples: int,
    seed: int,
) -> pl.DataFrame:
    """The elasticity of a day's tickets to the menu's price level, with weekday effects,
    from the shop's own price changes; its interval resamples whole weeks.

    Twice: over the whole day, and over the quiet hours alone - the weekday-and-hour slots
    whose orders seldom wait past the owner's target (90th percentile within it). At the
    rush, the queue turns customers away whatever the price, and hides part of the answer to
    it; in a quiet hour the answer is the customers'. Every price moved together: the
    history cannot tell one product's answer from another's."""
    slots = (
        orders.with_columns(pl.col("ordered_at").dt.weekday().alias("weekday"))
        .group_by("weekday", "hour")
        .agg(pl.col("minutes").quantile(0.9).alias("p90"))
    )
    quiet = slots.filter(pl.col("p90") <= wait_target).select("weekday", "hour")
    rows = [
        _elasticity("whole day", hours, resamples, seed),
        _elasticity("quiet hours", hours.join(quiet, on=["weekday", "hour"]), resamples, seed),
    ]
    return pl.DataFrame(rows, schema=PRICE_RESPONSE)


def _elasticity(scope: str, hours: pl.DataFrame, resamples: int, seed: int) -> dict[str, object]:
    days = (
        hours.group_by("date")
        .agg(pl.col("tickets").sum(), pl.col("price_level").first(), pl.col("weekday").first())
        .filter(pl.col("tickets") > 0)
        .sort("date")
    )
    levels = days["price_level"].unique()
    row: dict[str, object] = {
        "scope": scope,
        "days": days.height,
        "price_levels": levels.len(),
        "level_min": levels.min(),
        "level_max": levels.max(),
    }
    if levels.len() < 2:  # one price all along: nothing to answer
        return row | {"elasticity": None, "elasticity_low": None, "elasticity_high": None}
    y = np.log(days["tickets"].to_numpy())
    weekday = days["weekday"].to_numpy()
    x = np.column_stack(
        [np.ones(len(y)), np.log(days["price_level"].to_numpy())]
        + [(weekday == d).astype(float) for d in sorted(set(weekday))[1:]]
    )
    weeks = days["date"].dt.truncate("1w").to_numpy()
    blocks = [np.flatnonzero(weeks == week) for week in np.unique(weeks)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        chosen = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(np.linalg.lstsq(x[chosen], y[chosen], rcond=None)[0][1])
    return row | {
        "elasticity": float(np.linalg.lstsq(x, y, rcond=None)[0][1]),
        "elasticity_low": float(np.percentile(draws, 2.5)),
        "elasticity_high": float(np.percentile(draws, 97.5)),
    }


def price_scenarios(
    hours: pl.DataFrame,
    sales: pl.DataFrame,
    margins: pl.DataFrame,
    response: pl.DataFrame,
    settings: StudiesConfig,
) -> pl.DataFrame:
    """A day now - the last `baseline_weeks` weeks - and the same day with every price moved
    by each scenario's percent: tickets by the whole day's elasticity (constant, an
    assumption), revenue, and gross profit, with the range the elasticity's interval gives.
    Staff and rent do not move with a price; gross profit is before them."""
    whole = response.filter(pl.col("scope") == "whole day").row(0, named=True)
    if whole["elasticity"] is None:
        return pl.DataFrame(schema=SCENARIOS)
    last = hours["date"].max()
    assert isinstance(last, date)
    since = date.fromordinal(last.toordinal() - 7 * settings.baseline_weeks + 1)
    recent = hours.filter(pl.col("date") >= since)
    open_days = recent["date"].n_unique()
    latest_cost = margins.sort("month").group_by("product").agg(pl.col("unit_cost_mxn").last())
    window = sales.filter(pl.col("date") >= since).join(latest_cost, on="product", how="left")
    tickets = float(recent["tickets"].sum()) / open_days
    revenue = float(recent["revenue_mxn"].sum()) / open_days
    goods = float((window["quantity"] * window["unit_cost_mxn"]).sum()) / open_days
    level_now = float(hours.filter(pl.col("date") == last)["price_level"][0])
    gross_now = revenue - goods
    rows = []
    for change in [0.0, *settings.scenarios]:
        factor = 1 + change / 100
        # The share of today's tickets a price `factor` times today's keeps; each keeps its
        # revenue, moved by the factor, and its goods, which the price does not move.
        kept = {k: factor ** whole[k] for k in ("elasticity", "elasticity_low", "elasticity_high")}
        profit = {k: (revenue * factor - goods) * share for k, share in kept.items()}
        bounds = [profit["elasticity_low"], profit["elasticity_high"]]
        central = profit["elasticity"]
        level = level_now * factor
        rows.append(
            {
                "change_pct": change,
                "price_level": level,
                "within_history": whole["level_min"] - 1e-9 <= level <= whole["level_max"] + 1e-9,
                "tickets_per_day": tickets * kept["elasticity"],
                "revenue_per_day": revenue * factor * kept["elasticity"],
                "gross_profit_per_day": central,
                "gross_profit_low": min(bounds),
                "gross_profit_high": max(bounds),
                "versus_now_pct": 100 * (central / gross_now - 1),
                "versus_now_low_pct": 100 * (min(bounds) / gross_now - 1),
                "versus_now_high_pct": 100 * (max(bounds) / gross_now - 1),
            }
        )
    return pl.DataFrame(rows, schema=SCENARIOS)


# --- When to raise ----------------------------------------------------------------------


def price_alerts(
    margins: pl.DataFrame,
    shop: ShopConfig,
    outlook: pl.DataFrame | None,
    cpi: pl.DataFrame | None,
    indicator: str,
    months: int,
) -> pl.DataFrame:
    """Each product in the latest month against the margin its owner aims at - now, and
    in `months` months if green coffee reaches the top of its outlook's range and prices in
    general keep rising as in the last year - and the rise that would restore it.

    The coffee in a product moves with green coffee by its pass-through (the share of what
    roasted beans cost that green coffee moves); the rest of its cost, with the consumer
    price index. `raise` when the margin is below its target now; `watch` when it would fall
    below at the horizon; `ok` otherwise. Without the parent's outlook and index, the future
    is not judged: the alert reads `raise` or `ok` from today alone."""
    latest = margins.filter(pl.col("month") == margins["month"].max())
    beans_move = _outlook_change(outlook, indicator, months)
    general_move = _inflation_over(cpi, months)
    judged = beans_move is not None and general_move is not None
    through = max((shop.ingredients[name].pass_through for name in beans_of(shop)), default=0.0)
    rows = []
    for item in latest.iter_rows(named=True):
        cost_now = float(item["unit_cost_mxn"])
        coffee = (item["beans_share_pct"] or 0.0) / 100
        cost_then = cost_now * (
            coffee * (1 + through * (beans_move or 0.0) / 100)
            + (1 - coffee) * (1 + (general_move or 0.0) / 100)
        )
        price, target = float(item["price_mxn"]), shop.target_margins[item["category"]]
        margin_then = 100 * (1 - cost_then / price)
        status = (
            "raise"
            if item["margin_pct"] < target
            else "watch"
            if judged and margin_then < target
            else "ok"
        )
        rows.append(
            {
                "product": item["product"],
                "category": item["category"],
                "month": item["month"],
                "price_mxn": price,
                "unit_cost_mxn": cost_now,
                "margin_pct": item["margin_pct"],
                "target_pct": target,
                "raise_needed_pct": max(0.0, 100 * (cost_now / (1 - target / 100) / price - 1)),
                "horizon_months": months,
                "green_coffee_high_pct": beans_move,
                "inflation_pct": general_move,
                "margin_at_horizon_pct": margin_then if judged else None,
                "status": status,
            }
        )
    return pl.DataFrame(rows, schema=ALERTS).sort("status", "product")


ALERTS = {
    "product": pl.String,
    "category": pl.String,
    "month": pl.Date,
    "price_mxn": pl.Float64,
    "unit_cost_mxn": pl.Float64,
    "margin_pct": pl.Float64,
    "target_pct": pl.Float64,
    "raise_needed_pct": pl.Float64,  # every price up this much restores the target now
    "horizon_months": pl.Int64,
    "green_coffee_high_pct": pl.Float64,  # the top of the outlook's range, % from now
    "inflation_pct": pl.Float64,  # the last twelve months' pace, over the horizon
    "margin_at_horizon_pct": pl.Float64,
    "status": pl.String,  # raise, watch or ok
}


def _outlook_change(outlook: pl.DataFrame | None, indicator: str, months: int) -> float | None:
    if outlook is None:
        return None
    row = outlook.filter(pl.col("indicator") == indicator, pl.col("horizon_months") == months)
    return float(row["change_high_pct"][0]) if row.height else None


def _inflation_over(cpi: pl.DataFrame | None, months: int) -> float | None:
    if cpi is None or cpi.height < 13:
        return None
    index = cpi.sort("month")["index"]
    yearly = float(index[-1]) / float(index[-13])
    return 100 * (float(yearly ** (months / 12)) - 1)


# --- What inflation did ------------------------------------------------------------------


def inflation_impact(
    hours: pl.DataFrame,
    margins: pl.DataFrame,
    shifts: pl.DataFrame,
    cpi: pl.DataFrame | None,
) -> pl.DataFrame:
    """Month by month, against the shop's first month: its menu's level, what a mix of its
    products cost to make, and an hour of staff - in pesos of the day and, deflated by the
    consumer price index, in pesos of the first month. A level that falls in real terms is
    a price rise inflation took back."""
    month = pl.col("date").dt.truncate("1mo").alias("month")
    menu = hours.group_by(month).agg(pl.col("price_level").mean())
    made = margins.group_by("month").agg(
        ((pl.col("units") * pl.col("unit_cost_mxn")).sum() / pl.col("units").sum()).alias("cost"),
        (
            100
            * (pl.col("units") * pl.col("margin_mxn")).sum()
            / (pl.col("units") * pl.col("price_mxn")).sum()
        ).alias("gross_margin_pct"),
    )
    staff = shifts.group_by(month).agg(pl.col("hourly_wage_mxn").mean().alias("wage"))
    table = menu.join(made, on="month").join(staff, on="month", how="left").sort("month")
    first = table.row(0, named=True)
    table = table.with_columns(
        (pl.col("cost") / first["cost"]).alias("cost_index"),
        (pl.col("wage") / first["wage"]).alias("wage_index"),
    )
    if cpi is None:
        deflator = pl.lit(None, dtype=pl.Float64)
    else:
        table = table.join(cpi.rename({"index": "cpi"}), on="month", how="left")
        deflator = pl.col("cpi") / table["cpi"][0]
    return table.with_columns(deflator.alias("cpi_index")).select(
        "month",
        "price_level",
        (pl.col("price_level") / pl.col("cpi_index")).alias("real_price_level"),
        "cost_index",
        (pl.col("cost_index") / pl.col("cpi_index")).alias("real_cost_index"),
        "wage_index",
        "cpi_index",
        "gross_margin_pct",
    )


# --- Which products carry the menu --------------------------------------------------------


def menu_engineering(margins: pl.DataFrame, months: int) -> pl.DataFrame:
    """The menu over its last `months` months as menu engineering reads it: popular when a
    product sells at least 70% of an equal share; profitable when its margin in pesos is at
    least the menu's, weighted by what sells. Star (both), plowhorse (popular, thin), puzzle
    (rich, slow), dog (neither)."""
    last = margins["month"].unique().sort().tail(months).to_list()
    recent = margins.filter(pl.col("month").is_in(last))
    table = recent.group_by("product", "category").agg(
        pl.col("units").sum(),
        ((pl.col("units") * pl.col("margin_mxn")).sum() / pl.col("units").sum()).alias(
            "margin_mxn"
        ),
        ((pl.col("units") * pl.col("price_mxn")).sum()).alias("revenue_mxn"),
    )
    share = 100 * pl.col("units") / pl.col("units").sum()
    average = float((table["units"] * table["margin_mxn"]).sum()) / float(table["units"].sum())
    popular = share >= 100 * POPULAR_SHARE / table.height
    rich = pl.col("margin_mxn") >= average
    return table.with_columns(
        share.alias("mix_pct"),
        pl.when(popular & rich)
        .then(pl.lit("star"))
        .when(popular)
        .then(pl.lit("plowhorse"))
        .when(rich)
        .then(pl.lit("puzzle"))
        .otherwise(pl.lit("dog"))
        .alias("class"),
        (pl.col("margin_mxn") - average).alias("margin_vs_menu_mxn"),
    ).sort("units", descending=True)


# --- Where the hands fall short -----------------------------------------------------------


def staffing(
    hours: pl.DataFrame, orders: pl.DataFrame, shifts: pl.DataFrame, wait_target: float
) -> pl.DataFrame:
    """Each weekday-and-hour slot: its tickets, the hands on shift, tickets per hand, what
    staff costs per ticket, and how long orders took - the 90th percentile and the share
    past the owner's target. A slot that waits long has too few hands for its crowd; one
    with few tickets per hand, more than it needs."""
    wage = shifts.group_by(pl.col("date").dt.truncate("1mo").alias("month")).agg(
        pl.col("hourly_wage_mxn").mean()
    )
    slots = (
        hours.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .join(wage, on="month", how="left")
        .group_by("weekday", "hour")
        .agg(
            pl.col("tickets").mean().alias("tickets"),
            pl.col("baristas").mean().alias("staff"),
            (pl.col("baristas") * pl.col("hourly_wage_mxn")).sum().alias("staff_cost"),
            pl.col("tickets").sum().alias("all_tickets"),
        )
    )
    waits = (
        orders.with_columns(pl.col("ordered_at").dt.weekday().alias("weekday"))
        .group_by("weekday", "hour")
        .agg(
            pl.col("minutes").quantile(0.9).alias("wait_p90_minutes"),
            (100 * (pl.col("minutes") > wait_target).mean()).alias("over_target_pct"),
        )
    )
    return (
        slots.join(waits, on=["weekday", "hour"], how="left")
        .with_columns(
            # A slot that sold nothing has no cost per ticket; one with no hands, no rate.
            pl.when(pl.col("staff") > 0)
            .then(pl.col("tickets") / pl.col("staff"))
            .alias("tickets_per_staff_hour"),
            pl.when(pl.col("all_tickets") > 0)
            .then(pl.col("staff_cost") / pl.col("all_tickets"))
            .alias("staff_cost_per_ticket_mxn"),
            (pl.col("wait_p90_minutes") > wait_target).fill_null(False).alias("long_waits"),
        )
        .select(
            "weekday",
            "hour",
            "tickets",
            "staff",
            "tickets_per_staff_hour",
            "staff_cost_per_ticket_mxn",
            "wait_p90_minutes",
            "over_target_pct",
            "long_waits",
        )
        .sort("weekday", "hour")
    )


# --- The shop's corner of the city ---------------------------------------------------------


def neighbourhood(
    shop: ShopConfig,
    clean: Mapping[str, pl.DataFrame],
    purchases: pl.DataFrame,
    beans: Collection[str],
    settings: StudiesConfig,
) -> pl.DataFrame:
    """The shop's corner against its borough and the city, from the parent's tables: the
    coffee shops around it, how many an AGEB like its own would have, who lives there, and
    what it pays for its beans against what the city's roasters charge. A measure whose
    parent table is not built is left out."""
    rows: list[dict[str, object]] = []
    places = clean.get(settings.coffee_shops)
    if places is not None:
        coffee = places.filter(pl.col("kind") == COFFEE)
        distance = _metres(shop.latitude, shop.longitude)
        for radius in settings.radii_m:
            near = coffee.filter(distance <= radius).height
            rows.append(_row(f"coffee shops within {radius:g} m", near, None, None, "places"))
    zones, expected = clean.get(settings.zones), clean.get(settings.zone_expectations)
    if zones is not None:
        own = zones.filter(pl.col("zone_id") == shop.zone_id)
        borough = zones.filter(pl.col("borough_id") == own["borough_id"][0]) if own.height else own
        for measure, column, unit in (
            ("residents", "population", "people"),
            ("density", "people_per_km2", "people per km²"),
            ("schooling", "schooling_years", "years"),
            ("homes with internet", "internet_pct", "%"),
            ("coffee shops listed in the AGEB", "coffee_shops", "places"),
        ):
            rows.append(
                _row(
                    measure,
                    _first(own, column),
                    borough[column].median(),
                    zones[column].median(),
                    unit,
                )
            )
    if expected is not None:
        mine = expected.filter(pl.col("zone_id") == shop.zone_id)
        borough = (
            expected.filter(pl.col("borough_id") == mine["borough_id"][0]) if mine.height else mine
        )
        rows.append(
            _row(
                "coffee shops an AGEB like it would have",
                _first(mine, "held_out_prediction"),
                borough["held_out_prediction"].median(),
                expected["held_out_prediction"].median(),
                "places",
            )
        )
    offers = clean.get(settings.roaster_offers)
    bought = purchases.filter(pl.col("ingredient").is_in(list(beans)), pl.col("unit") == "g")
    if offers is not None and bought.height:
        latest = bought.filter(pl.col("date") == bought["date"].max())
        fair = offers.filter(
            ~pl.col("price_outlier"), pl.col("snapshot") == offers["snapshot"].max()
        )
        rows.append(
            _row(
                "beans, a kilogram",
                1000 * float(latest["unit_cost_mxn"][0]),
                None,
                fair["price_mxn_per_kg"].median(),
                "MXN (city: roasters' shelf price)",
            )
        )
    return pl.DataFrame(rows, schema=NEIGHBOURHOOD)


def _metres(latitude: float, longitude: float) -> pl.Expr:
    """Great-circle metres from a point to each row's `latitude`, `longitude`."""
    lat1, lon1 = math.radians(latitude), math.radians(longitude)
    lat2, lon2 = pl.col("latitude").radians(), pl.col("longitude").radians()
    a = ((lat2 - lat1) / 2).sin() ** 2 + math.cos(lat1) * lat2.cos() * (
        (lon2 - lon1) / 2
    ).sin() ** 2
    distance: pl.Expr = 2 * EARTH_RADIUS_M * a.sqrt().arcsin()
    return distance


def _first(frame: pl.DataFrame, column: str) -> object:
    return frame[column][0] if frame.height else None


def _row(measure: str, shop: object, borough: object, city: object, unit: str) -> dict[str, object]:
    return {"measure": measure, "shop": shop, "borough": borough, "city": city, "unit": unit}
