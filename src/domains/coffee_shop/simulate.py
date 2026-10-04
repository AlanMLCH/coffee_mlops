"""The demo shop's point-of-sale export, simulated: a neighbourhood specialty coffee shop
in Mexico City that does not exist, so that every model of this domain has data to learn
from - and anchored, number by number, to data that does exist.

What is real, and read at every simulation:
- **When people buy**: the share of a day's tickets in each hour and how each weekday
  compares, from a real coffee machine's sales (`sales_pattern`, CC0).
- **How they answer a price**: the elasticity of its daily sales to its own price changes
  (it moved every price several times in 2024), estimated with an interval. A vending
  machine's, and confounded with the season: said wherever it is used.
- **What the shop pays**: beans follow the green coffee price in pesos, milk, pastries and
  wages follow the consumer price index - both lent by the coffee domain.

What is assumed, in `config.yaml` (`shop`): the menu and its prices, recipes, opening
hours, shifts, wages, and how many tickets an ordinary day brings. What is simulated:
every ticket, its items and minute, how long it waited, the weekly purchases and the
shifts. One seed: the same anchors and config give the same export, byte for byte, and the
raw layer stores nothing new.

A queue decides the waits: each ticket is made by the first barista free, in the order it
came, and takes the sum of its items' minutes, each varied. A customer who would wait
longer than their patience leaves without buying, and the export never sees them - as a
real one would not. A shop short of hands at its rush shows it as waits and as sales that
stop growing where the crowd does, which is what an efficiency model has to find.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from domains.coffee_shop.config import ShopConfig, SimulationConfig
from mlops_core.adapter import ApiExtraction
from mlops_core.data.extract import latest_ingestion, store_payload

# The export's tables, each stored as a raw source of its own.
TABLES = ("pos_sales", "pos_orders", "pos_menu", "pos_purchases", "pos_shifts", "pos_recipes")
# How much a day varies around what it should be: a gamma with this shape has a
# coefficient of variation of one over its square root, about 22%.
DAY_SHAPE = 20.0
PREP_SPREAD = 0.25  # an item's minutes vary log-normally by this much
# A customer's patience for the queue, in minutes before a barista starts their order:
# exponential, so one in four leaves facing four minutes and half facing ten. Assumed.
PATIENCE_MINUTES = 15.0
DINE_IN, CARD = 0.45, 0.8
EXPORT_URL = "simulation://coffee_shop/pos-export"


@dataclass(frozen=True)
class Anchors:
    """What the real sales say, for a shop to borrow."""

    hour_shares: dict[int, float]  # hour -> share of a day's tickets
    weekday_factors: dict[int, float]  # ISO weekday -> its days against the mean day
    elasticity: float  # % change in units for a 1% change in price
    elasticity_low: float
    elasticity_high: float
    days: int  # how many days of sales it was read from


def sales_anchors(sales: pl.DataFrame, resamples: int = 1000, seed: int = 7) -> Anchors:
    """Hours, weekdays and the price elasticity of a real seller's sales.

    `sales` is the raw file as text: `date`, `datetime`, `money` (the price paid) and
    `coffee_name`. A product's price is set against what it cost in the first week; a
    day's price level is the mean of those ratios over its sales, carried over the days
    with none. The elasticity is the slope of log units on log price level over days,
    with weekday effects, and its interval comes from resampling days."""
    frame = sales.select(
        pl.col("date").str.to_date(),
        pl.col("datetime").str.to_datetime(),
        pl.col("money").cast(pl.Float64),
        pl.col("coffee_name"),
    )
    first, last = frame["date"].min(), frame["date"].max()
    assert isinstance(first, date) and isinstance(last, date)
    first_week = first + timedelta(days=7)
    reference = (
        frame.filter(pl.col("date") < first_week)
        .group_by("coffee_name")
        .agg(pl.col("money").median().alias("reference"))
    )
    priced = frame.join(reference, on="coffee_name", how="inner").with_columns(
        (pl.col("money") / pl.col("reference")).alias("ratio")
    )
    days = priced.group_by("date").agg(pl.len().alias("units"), pl.col("ratio").mean())
    calendar = pl.DataFrame({"date": pl.date_range(first, last, eager=True)})
    daily = (
        calendar.join(days, on="date", how="left")
        .sort("date")
        .with_columns(pl.col("ratio").forward_fill(), pl.col("units").fill_null(0))
        .with_columns(pl.col("date").dt.weekday().alias("weekday"))
    )
    hours = frame.group_by(pl.col("datetime").dt.hour().alias("hour")).len()
    weekdays = daily.group_by("weekday").agg(pl.col("units").mean())
    mean_day = float(daily["units"].mean())  # type: ignore[arg-type]
    y = np.log(daily["units"].to_numpy() + 1.0)
    weekday = daily["weekday"].to_numpy()
    x = np.column_stack(
        [np.ones(len(y)), np.log(daily["ratio"].to_numpy())]
        + [(weekday == day).astype(float) for day in range(2, 8)]
    )
    slope = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
    rng = np.random.default_rng(seed)
    draws = [
        np.linalg.lstsq(x[rows], y[rows], rcond=None)[0][1]
        for rows in (rng.integers(0, len(y), len(y)) for _ in range(resamples))
    ]
    return Anchors(
        hour_shares={
            int(h): float(n) / float(hours["len"].sum()) for h, n in hours.sort("hour").iter_rows()
        },
        weekday_factors={int(d): float(u) / mean_day for d, u in weekdays.iter_rows()},
        elasticity=slope,
        elasticity_low=float(np.percentile(draws, 2.5)),
        elasticity_high=float(np.percentile(draws, 97.5)),
        days=daily.height,
    )


def cost_factors(
    green: pl.DataFrame, cpi: pl.DataFrame, indicator: str, first_month: date
) -> dict[str, dict[date, float]]:
    """Month -> how much each followed cost has moved since the shop opened: the green
    coffee price in pesos (`green_coffee`) and the consumer price index (`inflation`). A
    month either has not published yet keeps the last one's."""
    beans = (
        green.filter(pl.col("indicator") == indicator)
        .select(pl.col("period").alias("month"), pl.col("mxn_per_kg").alias("value"))
        .sort("month")
    )
    index = cpi.select("month", pl.col("index").alias("value")).sort("month")
    factors = {}
    for name, series in (("green_coffee", beans), ("inflation", index)):
        base = series.filter(pl.col("month") <= first_month)["value"]
        if base.is_empty():
            raise ValueError(f"No {name} month at or before the shop opened ({first_month})")
        start = float(base[-1])
        factors[name] = {m: float(v) / start for m, v in series.iter_rows()}
    return factors


def factor_on(factors: Mapping[date, float], day: date) -> float:
    """The factor of the day's month, or of the latest month published before it."""
    month = day.replace(day=1)
    known = [m for m in factors if m <= month]
    return factors[max(known)] if known else 1.0


def price_level(shop: ShopConfig, day: date) -> float:
    """Every menu price, against opening day: the changes in force, compounded."""
    level = 1.0
    for change in shop.price_changes:
        if change.starts <= day and (change.until is None or day <= change.until):
            level *= 1 + change.change_pct / 100
    return level


def simulate(
    shop: ShopConfig, anchors: Anchors, factors: Mapping[str, Mapping[date, float]]
) -> dict[str, list[dict[str, Any]]]:
    """The shop's whole export, every table, from its first day to its last."""
    rng = np.random.default_rng(shop.seed)
    products = [item.product for item in shop.menu]
    shares = np.array([item.share for item in shop.menu])
    shares /= shares.sum()
    menu = {item.product: item for item in shop.menu}
    sales: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    used: dict[tuple[date, str], float] = {}
    shifts: list[dict[str, Any]] = []
    day = shop.first_day
    while day <= shop.last_day:
        weekday = day.isoweekday()
        if weekday in shop.opens:
            opens = datetime.combine(day, shop.opens[weekday])
            closes = datetime.combine(day, shop.closes[weekday])
            level = price_level(shop, day)
            wage = shop.hourly_wage_mxn * factor_on(factors["inflation"], day)
            staff = _on_shift(shop, day, wage, shifts)
            expected = (
                shop.tickets_per_day
                * anchors.weekday_factors.get(weekday, 1.0)
                * level**anchors.elasticity
                * rng.gamma(DAY_SHAPE, 1 / DAY_SHAPE)
            )
            tickets = _tickets(rng, anchors, opens, closes, expected)
            baristas_free = {name: opens for name in staff}
            sold = 0
            for ordered_at in tickets:
                working = [name for name, (start, end) in staff.items()
                           if start <= ordered_at < end] or list(staff)  # fmt: skip
                barista = min(working, key=lambda name: baristas_free[name])
                starts = max(ordered_at, baristas_free[barista])
                if (starts - ordered_at) / timedelta(minutes=1) > rng.exponential(PATIENCE_MINUTES):
                    continue  # left without buying
                sold += 1
                ticket = f"{day:%Y%m%d}-{sold:04d}"
                count = int(rng.choice(len(shop.items_per_ticket), p=shop.items_per_ticket)) + 1
                chosen = rng.choice(len(products), size=count, p=shares)
                minutes = 0.0
                for line, index in enumerate(chosen, start=1):
                    item = menu[products[int(index)]]
                    minutes += item.prep_minutes * float(rng.lognormal(0, PREP_SPREAD))
                    sales.append(
                        {"ticket": ticket, "line": line, "sold_at": ordered_at.isoformat(),
                         "product": item.product, "quantity": 1,
                         "unit_price": round(item.price_mxn * level, 2),
                         "channel": "dine_in" if rng.random() < DINE_IN else "takeaway",
                         "payment": "card" if rng.random() < CARD else "cash"}
                    )  # fmt: skip
                    for ingredient, quantity in item.recipe.items():
                        used[(day, ingredient)] = used.get((day, ingredient), 0.0) + quantity
                ready = starts + timedelta(minutes=minutes)
                baristas_free[barista] = ready
                orders.append(
                    {"ticket": ticket, "ordered_at": ordered_at.isoformat(),
                     "ready_at": ready.isoformat(timespec="seconds"), "items": count,
                     "baristas": len(working)}
                )  # fmt: skip
        day += timedelta(days=1)
    return {
        "pos_sales": sales,
        "pos_orders": orders,
        "pos_menu": _menu_history(shop),
        "pos_purchases": _purchases(shop, used, factors),
        "pos_shifts": shifts,
        "pos_recipes": [
            {
                "product": item.product,
                "ingredient": name,
                "quantity": quantity,
                "unit": shop.ingredients[name].unit,
            }
            for item in shop.menu
            for name, quantity in item.recipe.items()
        ],
    }


def _tickets(
    rng: np.random.Generator,
    anchors: Anchors,
    opens: datetime,
    closes: datetime,
    expected: float,
) -> list[datetime]:
    """A day's tickets, in time order: each open hour gets its share of the day (only the
    part of it the shop is open), and its tickets fall at random minutes in it."""
    hours = []
    hour = opens.replace(minute=0, second=0)
    while hour < closes:
        start, end = max(hour, opens), min(hour + timedelta(hours=1), closes)
        open_share = (end - start) / timedelta(hours=1)
        hours.append((start, end, anchors.hour_shares.get(hour.hour, 0.0) * open_share))
        hour += timedelta(hours=1)
    total = sum(weight for _, _, weight in hours) or 1.0
    tickets = []
    for start, end, weight in hours:
        seconds = (end - start).total_seconds()
        for _ in range(int(rng.poisson(expected * weight / total))):
            tickets.append(start + timedelta(seconds=float(rng.uniform(0, seconds))))
    return sorted(t.replace(microsecond=0) for t in tickets)


def _on_shift(
    shop: ShopConfig, day: date, wage: float, shifts: list[dict[str, Any]]
) -> dict[str, tuple[datetime, datetime]]:
    """The staff working `day`, each with their hours; their shifts are recorded."""
    staff = {}
    for n, shift in enumerate(shop.shifts, start=1):
        if day.isoweekday() not in shift.days:
            continue
        name = f"{shift.role}-{n}"
        start, end = datetime.combine(day, shift.starts), datetime.combine(day, shift.ends)
        staff[name] = (start, end)
        shifts.append(
            {"date": day.isoformat(), "staff": name, "role": shift.role,
             "starts_at": start.isoformat(), "ends_at": end.isoformat(),
             "hourly_wage": round(wage, 2)}
        )  # fmt: skip
    return staff


def _menu_history(shop: ShopConfig) -> list[dict[str, Any]]:
    """Each product's price, period by period: a new period wherever a change starts or a
    promotion ends."""
    edges = {shop.first_day}
    for change in shop.price_changes:
        edges.add(change.starts)
        if change.until is not None:
            edges.add(change.until + timedelta(days=1))
    bounds = sorted(edge for edge in edges if shop.first_day <= edge <= shop.last_day)
    rows = []
    for start, nxt in zip(bounds, [*bounds[1:], None], strict=True):
        level = price_level(shop, start)
        for item in shop.menu:
            rows.append(
                {"product": item.product, "category": item.category,
                 "price": round(item.price_mxn * level, 2), "valid_from": start.isoformat(),
                 "valid_to": (nxt - timedelta(days=1)).isoformat() if nxt else None}
            )  # fmt: skip
    return rows


def _purchases(
    shop: ShopConfig, used: Mapping[tuple[date, str], float], factors: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """What the shop buys each Monday for its week: the week's use and its waste, at
    that month's cost."""
    weekly: dict[tuple[date, str], float] = {}
    for (day, name), quantity in used.items():
        monday = day - timedelta(days=day.weekday())
        weekly[(monday, name)] = weekly.get((monday, name), 0.0) + quantity
    rows = []
    for (monday, name), quantity in sorted(weekly.items()):
        stock = shop.ingredients[name]
        bought = quantity * (1 + stock.waste_pct / 100)
        moved = factor_on(factors[stock.follows], monday) if stock.follows else 1.0
        factor = 1 + stock.pass_through * (moved - 1)
        rows.append(
            {"date": monday.isoformat(), "ingredient": name, "unit": stock.unit,
             "quantity": round(bought, 1), "cost": round(bought * stock.cost_mxn * factor, 2)}
        )  # fmt: skip
    return rows


def extract(
    shop: ShopConfig,
    simulation: SimulationConfig,
    data_dir: Path,
    lent: Mapping[str, pl.DataFrame],
    now: datetime | None = None,
) -> ApiExtraction:
    """Simulate the export from the newest real anchors and store each table in raw.

    Without the real sales or the lent tables there is nothing to anchor to, and the
    export is skipped out loud: a shop made of assumptions alone is not this one."""
    raw_dir = data_dir / "raw"
    pattern = latest_ingestion(raw_dir, simulation.sales_pattern)
    missing = [name for name in (simulation.green_coffee, simulation.inflation) if name not in lent]
    if pattern is None or missing:
        why = "the real sales pattern is not downloaded" if pattern is None else f"no {missing}"
        return ApiExtraction(skipped=dict.fromkeys(TABLES, f"{why}: run the coffee domain first"))
    anchors = sales_anchors(pl.read_csv(pattern.path, infer_schema_length=0))
    factors = cost_factors(
        lent[simulation.green_coffee],
        lent[simulation.inflation],
        simulation.green_indicator,
        shop.first_day.replace(day=1),
    )
    export = simulate(shop, anchors, factors)
    stamp = now or datetime.now(UTC)
    artifacts = {}
    for table, rows in export.items():
        document = {"anchors": asdict(anchors), "rows": rows}
        body = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
        artifacts[table] = store_payload(table, f"{table}.json", body, raw_dir, EXPORT_URL, stamp)
    return ApiExtraction(artifacts=artifacts)


def to_frame(document: dict[str, Any]) -> pl.DataFrame:
    """A stored table's rows as text, the way any export would arrive."""
    rows = document["rows"]
    if not rows:
        return pl.DataFrame()
    columns = list(rows[0])
    return pl.DataFrame(
        {c: [None if r[c] is None else str(r[c]) for r in rows] for c in columns},
        schema=dict.fromkeys(columns, pl.String),
    )
