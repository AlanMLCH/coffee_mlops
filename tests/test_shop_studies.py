"""What a coffee shop's studies say for its owner, on data written so the answer can be
worked out by hand: margins, the price response and its scenarios, when to raise, what
inflation did, the menu, the hands, and the shop's corner of the city."""

import re
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

import domains.coffee.shops
from domains.coffee.shops import analysis
from domains.coffee.shops.adapter import CoffeeShopAdapter
from domains.coffee.shops.analysis import (
    inflation_impact,
    menu_engineering,
    neighbourhood,
    price_alerts,
    price_response,
    price_scenarios,
    product_margins,
    staffing,
)
from domains.coffee.shops.clean import clean_shop
from domains.coffee.shops.config import ShopConfig
from domains.coffee.shops.features import add_order_context
from domains.coffee.shops.request import ShopOrder
from domains.coffee.shops.simulate import simulate
from tests.test_coffee_shop import ANCHORS, FLAT, export_frames, two_weeks

JAN, FEB = date(2025, 1, 1), date(2025, 2, 1)


@pytest.fixture
def shop_adapter() -> CoffeeShopAdapter:
    return domains.coffee.shops.adapter("cafe_de_barrio")


def sales_of(rows: list[tuple[date, str, str, int, float]]) -> pl.DataFrame:
    """(day, product, category, quantity, unit price) -> sales lines."""
    return pl.DataFrame(
        [
            {"date": d, "product": p, "category": c, "quantity": q, "line_total_mxn": q * price}
            for d, p, c, q, price in rows
        ],
        schema={
            "date": pl.Date,
            "product": pl.String,
            "category": pl.String,
            "quantity": pl.Int64,
            "line_total_mxn": pl.Float64,
        },
    )


RECIPES = pl.DataFrame(
    {
        "product": ["latte", "latte", "concha"],
        "ingredient": ["coffee_beans", "milk", "concha"],
        "quantity": [18.0, 200.0, 1.0],
        "unit": ["g", "ml", "piece"],
    }
)


def test_a_product_costs_what_went_into_it_and_what_was_thrown_away() -> None:
    sales = sales_of(
        [
            (JAN + timedelta(days=2), "latte", "milk", 10, 70.0),
            (FEB + timedelta(days=2), "latte", "milk", 10, 77.0),
            (JAN + timedelta(days=2), "concha", "pastry", 10, 20.0),
        ]
    )
    # A tenth more bought than the recipes used, in January only: February keeps its costs.
    purchases = pl.DataFrame(
        {
            "date": [JAN, JAN, JAN],
            "ingredient": ["coffee_beans", "milk", "concha"],
            "unit": ["g", "ml", "piece"],
            "quantity": [396.0, 4400.0, 11.0],
            "cost_mxn": [178.2, 123.2, 88.0],
        }
    )

    margins = product_margins(sales, RECIPES, purchases, {"coffee_beans"})

    latte = margins.filter(pl.col("product") == "latte").sort("month")
    cost = 18 * 0.45 * 1.1 + 200 * 0.028 * 1.1  # 8.91 of coffee and 6.16 of milk
    assert latte["unit_cost_mxn"].to_list() == pytest.approx([cost, cost])
    assert latte["margin_pct"].to_list() == pytest.approx(
        [100 * (1 - cost / 70), 100 * (1 - cost / 77)]
    )
    assert latte["beans_share_pct"][0] == pytest.approx(100 * 8.91 / cost)
    concha = margins.filter(pl.col("product") == "concha").row(0, named=True)
    assert concha["unit_cost_mxn"] == pytest.approx(8 * 1.1)
    assert concha["beans_share_pct"] == 0


def hours_at(levels: list[float], weeks_each: int = 4, base: float = 100.0) -> pl.DataFrame:
    """Two open hours a day, each the half of a day whose tickets are `base` at level 1 and
    answer the level with an elasticity of exactly -1; Sundays a fifth busier."""
    rows, day = [], date(2025, 1, 6)
    for level in levels:
        for _ in range(7 * weeks_each):
            weekday = day.isoweekday()
            day_tickets = base / level * (1.2 if weekday == 7 else 1.0)
            for hour in (9, 17):
                rows.append(
                    {
                        "date": day,
                        "hour": hour,
                        "weekday": weekday,
                        "tickets": day_tickets / 2,
                        "price_level": level,
                        "revenue_mxn": day_tickets / 2 * 60 * level,
                        "baristas": 1,
                    }
                )
            day += timedelta(days=1)
    return pl.DataFrame(rows)


def orders_for(hours: pl.DataFrame, slow_hour: int, minutes: float = 3.0) -> pl.DataFrame:
    """An order a ticket, `minutes` long, or three times that in `slow_hour`."""
    return hours.select(
        (pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=pl.col("hour"))).alias(
            "ordered_at"
        ),
        "date",
        "hour",
        pl.when(pl.col("hour") == slow_hour).then(3 * minutes).otherwise(minutes).alias("minutes"),
    )


def test_the_price_response_recovers_an_elasticity_written_into_the_sales() -> None:
    hours = hours_at([1.0, 1.1, 0.95])

    response = price_response(hours, orders_for(hours, slow_hour=9), 8.0, 200, 0)

    whole, quiet = response.rows(named=True)
    assert whole["scope"] == "whole day" and quiet["scope"] == "quiet hours"
    assert whole["elasticity"] == pytest.approx(-1.0, abs=1e-6)
    assert whole["elasticity_low"] <= -1.0 + 1e-6 and whole["elasticity_high"] >= -1.0 - 1e-6
    assert (whole["price_levels"], whole["level_min"], whole["level_max"]) == (3, 0.95, 1.1)
    assert quiet["elasticity"] == pytest.approx(-1.0, abs=1e-6)  # the 17:00 hours alone
    one_price = price_response(hours_at([1.0]), orders_for(hours_at([1.0]), 9), 8.0, 200, 0)
    assert one_price["elasticity"].to_list() == [None, None]


def test_a_price_scenario_moves_tickets_by_the_elasticity_and_keeps_the_goods_cost(
    shop_adapter: CoffeeShopAdapter,
) -> None:
    hours = hours_at([1.0, 1.1])
    response = price_response(hours, orders_for(hours, 9), 8.0, 200, 0)
    last_days = hours.filter(pl.col("date") >= hours["date"].max() - timedelta(days=55))
    sales = sales_of([(d, "latte", "milk", 1, 66.0) for d in last_days["date"].unique()])
    margins = pl.DataFrame({"month": [JAN], "product": ["latte"], "unit_cost_mxn": [10.0]})
    settings = shop_adapter.config.studies.model_copy(update={"scenarios": [10.0, -5.0]})

    scenarios = price_scenarios(hours, sales, margins, response, settings)

    now, up, down = scenarios.rows(named=True)
    days = last_days["date"].n_unique()
    tickets = last_days["tickets"].sum() / days
    revenue, goods = last_days["revenue_mxn"].sum() / days, 10.0  # one latte a day
    assert now["change_pct"] == 0 and now["versus_now_pct"] == pytest.approx(0)
    assert up["tickets_per_day"] == pytest.approx(tickets / 1.1)
    assert up["gross_profit_per_day"] == pytest.approx((revenue * 1.1 - goods) / 1.1)
    assert up["within_history"] is False and down["within_history"] is True
    assert up["gross_profit_low"] <= up["gross_profit_per_day"] <= up["gross_profit_high"]
    flat = response.with_columns(pl.lit(None, dtype=pl.Float64).alias("elasticity"))
    assert price_scenarios(hours, sales, margins, flat, settings).is_empty()


def latest_margins(**rows: tuple[str, float, float, float]) -> pl.DataFrame:
    """product -> (category, price, unit cost, coffee's share of the cost %)."""
    return pl.DataFrame(
        [
            {
                "month": FEB,
                "product": p,
                "category": c,
                "units": 10,
                "price_mxn": price,
                "unit_cost_mxn": cost,
                "margin_mxn": price - cost,
                "margin_pct": 100 * (1 - cost / price),
                "beans_share_pct": coffee,
            }
            for p, (c, price, cost, coffee) in rows.items()
        ]
    )


def test_a_price_alert_says_raise_now_watch_ahead_or_ok(shop_adapter: CoffeeShopAdapter) -> None:
    shop = shop_adapter.config.shop  # espresso aims at 78%, pastry at 55%; beans pass half
    margins = latest_margins(
        espresso=("espresso", 40.0, 10.0, 100.0),  # 75%: below its 78% now
        americano=("espresso", 50.0, 10.5, 100.0),  # 79% now; beans up 40%, half passed: 74.8%
        croissant=("pastry", 50.0, 20.0, 0.0),  # 60%, and only inflation moves it
    )
    outlook = pl.DataFrame(
        {"indicator": ["other_milds"], "horizon_months": [6], "change_high_pct": [40.0]}
    )
    cpi = pl.DataFrame(
        {"month": [date(2024, m, 1) for m in range(1, 13)] + [JAN], "index": [100.0] * 12 + [104.0]}
    )

    alerts = {
        row["product"]: row
        for row in price_alerts(margins, shop, outlook, cpi, "other_milds", 6).rows(named=True)
    }

    assert alerts["espresso"]["status"] == "raise"
    assert alerts["espresso"]["raise_needed_pct"] == pytest.approx(100 * (10 / 0.22 / 40 - 1))
    assert alerts["americano"]["status"] == "watch"
    assert alerts["americano"]["margin_at_horizon_pct"] == pytest.approx(
        100 * (1 - 10.5 * 1.2 / 50)
    )
    assert alerts["croissant"]["status"] == "ok"
    inflation = 100 * (1.04**0.5 - 1)
    assert alerts["croissant"]["inflation_pct"] == pytest.approx(inflation)
    assert alerts["croissant"]["margin_at_horizon_pct"] == pytest.approx(
        100 * (1 - 20 * (1 + inflation / 100) / 50)
    )
    blind = price_alerts(margins, shop, None, None, "other_milds", 6)
    assert set(blind["status"]) == {"raise", "ok"}
    assert blind["margin_at_horizon_pct"].null_count() == blind.height


def test_inflation_shows_what_a_price_rise_kept_in_real_terms() -> None:
    hours = pl.DataFrame({"date": [JAN, FEB], "price_level": [1.0, 1.1]})
    margins = pl.DataFrame(
        {
            "month": [JAN, FEB],
            "units": [10, 10],
            "unit_cost_mxn": [20.0, 22.0],
            "margin_mxn": [50.0, 55.0],
            "price_mxn": [70.0, 77.0],
        }
    )
    shifts = pl.DataFrame({"date": [JAN, FEB], "hourly_wage_mxn": [75.0, 78.0]})
    cpi = pl.DataFrame({"month": [JAN, FEB], "index": [140.0, 147.0]})

    impact = inflation_impact(hours, margins, shifts, cpi).row(1, named=True)

    assert impact["real_price_level"] == pytest.approx(1.1 / 1.05)
    assert impact["real_cost_index"] == pytest.approx(1.1 / 1.05)
    assert impact["wage_index"] == pytest.approx(78 / 75)
    without = inflation_impact(hours, margins, shifts, None)
    assert without["real_price_level"].null_count() == 2


def test_menu_engineering_sorts_the_menu_into_four_kinds() -> None:
    margins = latest_margins(
        latte=("milk", 80.0, 20.0, 40.0),  # sells and earns: star
        americano=("espresso", 50.0, 10.0, 70.0),  # sells, earns less: plowhorse
        filter_v60=("filter", 90.0, 15.0, 60.0),  # earns, sells little: puzzle
        espresso=("espresso", 40.0, 10.0, 70.0),  # neither: dog
    ).with_columns(
        pl.col("product")
        .replace_strict({"latte": 40, "americano": 40, "filter_v60": 5, "espresso": 5})
        .alias("units")
    )

    menu = {row["product"]: row["class"] for row in menu_engineering(margins, 3).rows(named=True)}

    assert menu == {
        "latte": "star",
        "americano": "plowhorse",
        "filter_v60": "puzzle",
        "espresso": "dog",
    }


def test_staffing_finds_the_slots_short_of_hands() -> None:
    hours = pl.DataFrame(
        {
            "date": [JAN, JAN, JAN],
            "hour": [8, 9, 10],
            "weekday": [3, 3, 3],
            "tickets": [20.0, 2.0, 0.0],
            "baristas": [1, 2, 0],
        }
    )
    shifts = pl.DataFrame({"date": [JAN], "hourly_wage_mxn": [75.0]})
    orders = pl.DataFrame(
        {
            "ordered_at": [datetime(2025, 1, 1, 8, 5), datetime(2025, 1, 1, 9, 5)],
            "hour": [8, 9],
            "minutes": [14.0, 3.0],
        }
    )

    slots = {row["hour"]: row for row in staffing(hours, orders, shifts, 8.0).rows(named=True)}

    assert slots[8]["long_waits"] and not slots[9]["long_waits"]
    assert slots[8]["tickets_per_staff_hour"] == 20 and slots[9]["staff_cost_per_ticket_mxn"] == 75
    assert slots[10]["tickets_per_staff_hour"] is None  # nobody on shift: no rate
    assert slots[10]["staff_cost_per_ticket_mxn"] is None  # nothing sold: no cost per ticket


def test_the_neighbourhood_is_read_from_the_parent_tables_the_shop_lists(
    shop_adapter: CoffeeShopAdapter,
) -> None:
    config = shop_adapter.config
    shop, settings = config.shop, config.studies
    near = shop.latitude + 0.002  # about 222 m north
    far = shop.latitude + 0.008  # about 890 m
    clean: dict[str, pl.DataFrame] = {
        # The registers side by side: OSM's twin of DENUE's near shop is the same place.
        settings.coffee_shops: pl.DataFrame(
            {
                "shop_id": ["denue-1", "osm-1", "denue-2", "osm-2", "denue-3"],
                "source": ["denue", "osm", "denue", "osm", "denue"],
                "matched_shop_id": ["osm-1", "denue-1", None, None, None],
                "kind": ["coffee", "coffee", "coffee", "coffee", "juice"],
                "latitude": [near, near, far, far, near],
                "longitude": [shop.longitude] * 5,
            }
        ),
        settings.zones: pl.DataFrame(
            {
                "zone_id": [shop.zone_id, "z2", "z3"],
                "borough_id": ["b", "b", "c"],
                "population": [3000.0, 1000.0, 5000.0],
                "people_per_km2": [1.0, 2.0, 3.0],
                "schooling_years": [12.0, 10.0, 14.0],
                "internet_pct": [90.0, 70.0, 50.0],
                "coffee_shops": [0.0, 2.0, 4.0],
            }
        ),
        settings.zone_expectations: pl.DataFrame(
            {
                "zone_id": [shop.zone_id, "z2", "z3"],
                "borough_id": ["b", "b", "c"],
                "held_out_prediction": [6.0, 2.0, 1.0],
            }
        ),
    }

    rows = {row["measure"]: row for row in neighbourhood(shop, clean, settings).rows(named=True)}

    # Each place once: the near shop both registers list counts one, the far two are two.
    assert [rows[f"coffee shops within {r:g} m"]["shop"] for r in settings.radii_m] == [1, 1, 3]
    assert (rows["residents"]["shop"], rows["residents"]["borough"], rows["residents"]["city"]) == (
        3000.0,
        2000.0,
        3000.0,
    )
    expected = rows["coffee shops an AGEB like it would have"]
    assert (expected["shop"], expected["borough"], expected["city"]) == (6.0, 4.0, 2.0)
    assert not any("beans" in measure for measure in rows)  # a wholesale price is no neighbour's
    assert neighbourhood(shop, {}, settings).is_empty()


def test_what_is_left_each_month_is_every_cost_taken_from_the_sales(
    shop_adapter: CoffeeShopAdapter,
) -> None:
    """Rent and services are given in pesos of one month and moved with the index to the
    others; the card terminal keeps its fee of the card sales only."""
    shop = shop_adapter.config.shop.model_copy(
        update={
            "rent_mxn": 1100.0,
            "services_mxn": 110.0,
            "fixed_costs_as_of": FEB,
            "card_fee_pct": 4.0,
        }
    )
    sales = pl.DataFrame(
        {
            "date": [JAN, JAN, FEB],
            "line_total_mxn": [1000.0, 500.0, 2000.0],
            "payment": ["card", "cash", "card"],
        }
    )
    margins = pl.DataFrame({"month": [JAN, FEB], "units": [10, 10], "unit_cost_mxn": [20.0, 30.0]})
    shifts = pl.DataFrame({"date": [JAN, FEB], "cost_mxn": [100.0, 150.0]})
    cpi = pl.DataFrame({"month": [JAN, FEB], "index": [100.0, 110.0]})

    jan, feb = analysis.monthly_results(sales, margins, shifts, shop, cpi).rows(named=True)

    assert jan["card_fees_mxn"] == pytest.approx(40.0) and jan["rent_mxn"] == pytest.approx(1000.0)
    assert jan["services_mxn"] == pytest.approx(100.0)
    assert jan["operating_profit_mxn"] == pytest.approx(1500 - 200 - 40 - 100 - 1000 - 100)
    assert jan["operating_margin_pct"] == pytest.approx(4.0)
    assert jan["prime_cost_pct"] == pytest.approx(20.0)
    assert (feb["rent_mxn"], feb["operating_profit_mxn"]) == (
        pytest.approx(1100.0),
        pytest.approx(260.0),
    )
    assert feb["prime_cost_pct"] == pytest.approx(22.5)
    # No index: as given. An index that starts later: its first month stands for before.
    plain = analysis.monthly_results(sales, margins, shifts, shop, None)
    assert plain["rent_mxn"].to_list() == [1100.0, 1100.0]
    later = analysis.monthly_results(
        sales, margins, shifts, shop, cpi.filter(pl.col("month") == FEB)
    )
    assert later["rent_mxn"].to_list() == [pytest.approx(1100.0)] * 2


def test_the_sales_outlook_is_judged_on_past_months_before_it_is_trusted(
    shop_adapter: CoffeeShopAdapter,
) -> None:
    """A steady crowd: every weekday's average day, times the days a month opens. Its past
    forecasts miss nothing, and repeating last month misses by the calendar alone."""
    config = shop_adapter.config
    shop, settings = config.shop, config.studies  # open every day of the week
    hours = hours_at(
        [1.0], weeks_each=26
    )  # 6 Jan to 6 Jul 2025: 100 tickets, 6,000 MXN; Sundays +20%

    outlook, backtest = analysis.sales_outlook(hours, shop, settings)

    august = outlook.row(0, named=True)
    assert (august["month"], august["horizon_months"], august["days_open"]) == (
        date(2025, 8, 1), 1, 31,
    )  # fmt: skip
    assert august["tickets"] == pytest.approx(26 * 100 + 5 * 120)  # five Sundays
    assert august["revenue_mxn"] == pytest.approx(26 * 6000 + 5 * 7200)
    assert august["revenue_low_mxn"] == pytest.approx(august["revenue_mxn"])  # it never missed
    assert outlook.height == settings.sales_months
    assert backtest.height == 6 and backtest["origin"].min() == date(2025, 3, 1)
    assert backtest["error_pct"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert backtest["naive_error_pct"].abs().max() > 1  # a 31-day month for a 30-day one

    short, judged = analysis.sales_outlook(hours_at([1.0], weeks_each=12), shop, settings)
    assert judged.height < analysis.MIN_MISSES and short["revenue_low_mxn"].null_count() == 12


def test_every_study_runs_on_a_simulated_shop(shop_adapter: CoffeeShopAdapter) -> None:
    shop = two_weeks(shop_adapter.config.shop)
    adapter = CoffeeShopAdapter(shop_adapter.config.model_copy(update={"shop": shop}))
    clean = clean_shop(export_frames(simulate(shop, ANCHORS, FLAT)), shop)

    tables = adapter.studies(clean)

    assert set(tables) == {
        "profile",
        "product_margins",
        "price_response",
        "price_scenarios",
        "price_alerts",
        "inflation_impact",
        "menu_engineering",
        "staffing",
        "neighbourhood",
        "monthly_results",
        "sales_outlook",
        "sales_backtest",
    }
    assert tables["price_response"]["price_levels"].to_list() == [1, 1]  # one price: no answer
    assert tables["price_scenarios"].is_empty()
    assert tables["neighbourhood"].is_empty()  # no parent table given
    assert tables["product_margins"]["product"].n_unique() == len(shop.menu)
    assert analysis.beans_of(shop) == {"coffee_beans"}
    dictionary = (domains.coffee.shops.SHOPS_DIR / "data_dictionary.md").read_text(encoding="utf-8")
    for name, table in tables.items():
        assert f"## `analysis.{name}`" in dictionary
        assert [c for c in table.columns if f"`{c}`" not in dictionary] == []


def test_an_order_knows_its_hours_crowd_in_batch_and_is_told_it_online(
    shop_adapter: CoffeeShopAdapter,
) -> None:
    hours = pl.DataFrame({"date": [JAN], "hour": [10], "tickets": [14.0]})
    batch = pl.DataFrame({"date": [JAN], "hour": [10], "weekday": [3], "items": [2]})
    asked = ShopOrder(weekday=6, hour=10, items=2, baristas=1, tickets_in_hour=20)

    known = add_order_context(batch, {"shop_hours": hours})
    told = shop_adapter.enrich(
        "order_minutes", pl.DataFrame([asked.to_item()]), {"shop_hours": hours}
    )

    assert known["tickets_in_hour"].to_list() == [14.0]
    assert known["weekday_hour"].to_list() == ["3-10"]
    assert told["tickets_in_hour"].to_list() == [20.0] and told["weekday_hour"].to_list() == [
        "6-10"
    ]
    assert shop_adapter.context_tables("order_minutes") == ("shop_hours",)
    assert shop_adapter.request_model("order_minutes") is ShopOrder
    example = shop_adapter.config.model_named("order_minutes").example
    item = ShopOrder.model_validate(example).to_item()
    assert item["weekday"] == item["date"].isoweekday() == 6  # the next Saturday


def test_a_shop_says_what_it_aims_at_and_who_is_on_shift(shop_adapter: CoffeeShopAdapter) -> None:
    shop: dict[str, Any] = shop_adapter.config.shop.model_dump(mode="json")

    with pytest.raises(
        ValidationError, match=r"No target margin for the menu's categories \['pastry'\]"
    ):
        ShopConfig.model_validate(
            {
                **shop,
                "target_margins": {
                    k: v for k, v in shop["target_margins"].items() if k != "pastry"
                },
            }
        )
    with pytest.raises(ValidationError, match="between 0 and 100"):
        ShopConfig.model_validate(
            {**shop, "target_margins": {**shop["target_margins"], "milk": 120}}
        )
    with pytest.raises(ValidationError, match="Open with nobody on shift"):
        ShopConfig.model_validate({**shop, "shifts": shop["shifts"][:1]})


def test_a_shops_explorer_reads_its_own_tables_and_the_parents_it_lists() -> None:
    """Every query the explorer holds - numbers, layers, findings, tables - names the
    shop's own tables or the coffee tables it lists, and nothing else: the page cannot
    show another shop's data, whatever its YAML says."""
    for name in domains.coffee.shops.businesses():
        config = domains.coffee.shops.shop_config(name)
        assert config.explore is not None
        assert config.explore.title == config.shop.name
        assert (config.explore.view.latitude, config.explore.view.longitude) == (
            config.shop.latitude,
            config.shop.longitude,
        )
        named = {
            table
            for query in config.explore.queries
            for table in re.findall(
                r"(?<![\w.])(?:\w+\.)?(?:clean|features|predictions|analysis)\.\w+", query
            )
        }
        lent = {table for table in named if table.count(".") == 2}
        assert lent <= config.parent_tables, f"{name} reads {sorted(lent - config.parent_tables)}"
        assert any(table.startswith("coffee.") for table in named)  # it is set against the city
        own = {table.split(".")[1] for table in named - lent if table.startswith("clean.")}
        assert own <= set(config_tables(name)), own


def config_tables(name: str) -> list[str]:
    return list(domains.coffee.shops.adapter(name).clean_contracts())
