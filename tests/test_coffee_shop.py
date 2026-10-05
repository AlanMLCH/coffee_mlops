"""The coffee shops: subdomains of coffee, a business each, that simulate their
point-of-sale exports from real anchors, clean them into canonical tables and ask their
models about their own hours."""

import io
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from pydantic import ValidationError

import domains.coffee.shops
from domains.coffee.shops.adapter import CoffeeShopAdapter
from domains.coffee.shops.clean import clean_menu, clean_shop, shop_hours
from domains.coffee.shops.config import CoffeeShopConfig, Shift, ShopConfig
from domains.coffee.shops.features import add_hour_context
from domains.coffee.shops.request import ShopHour
from domains.coffee.shops.schemas import CLEAN_SCHEMAS
from domains.coffee.shops.simulate import (
    TABLES,
    Anchors,
    _menu_history,
    cost_factors,
    factor_on,
    price_level,
    sales_anchors,
    simulate,
    to_frame,
)
from mlops_core.adapter import domain_dir
from mlops_core.contracts import check_contract
from mlops_core.data.clean import build_clean
from mlops_core.data.extract import store_payload
from mlops_core.storage import write_table

# Every hour the same share, every weekday the same: what a test can count on.
ANCHORS = Anchors(
    hour_shares={hour: 1 / 16 for hour in range(7, 23)},
    weekday_factors=dict.fromkeys(range(1, 8), 1.0),
    elasticity=-1.0,
    elasticity_low=-1.5,
    elasticity_high=-0.5,
    days=60,
)
FLAT = {"green_coffee": {date(2025, 1, 1): 1.0}, "inflation": {date(2025, 1, 1): 1.0}}


@pytest.fixture
def shop_adapter() -> CoffeeShopAdapter:
    return domains.coffee.shops.adapter("cafe_de_barrio")


def two_weeks(shop: ShopConfig, **update: Any) -> ShopConfig:
    """The YAML's shop, open 1-14 January 2025: a Wednesday to a Tuesday."""
    return shop.model_copy(
        update={"first_day": date(2025, 1, 1), "last_day": date(2025, 1, 14), **update}
    )


def export_frames(export: dict[str, list[dict[str, Any]]]) -> dict[str, pl.DataFrame]:
    """The export as the core hands it to the domain: each table read from its JSON."""
    return {table: to_frame({"anchors": {}, "rows": rows}) for table, rows in export.items()}


def vending_csv(days: int = 56) -> bytes:
    """A seller's sales, written: four weeks a latte at 10, then four at 12, and the daily
    units falling from 60 to 50 - an elasticity of log(50/60)/log(1.2), about -1."""
    lines = ["date,datetime,cash_type,card,money,coffee_name"]
    for n in range(days):
        day = date(2024, 3, 4) + timedelta(days=n)
        price, units = (10.0, 60) if n < days // 2 else (12.0, 50)
        for unit in range(units):
            hour = 8 if unit % 2 else 15
            lines.append(f"{day},{day} {hour:02d}:{unit % 60:02d}:00.000,card,,{price},Latte")
    return "\n".join(lines).encode()


def test_every_shop_is_a_file_and_shares_the_rest(shop_adapter: CoffeeShopAdapter) -> None:
    barrio, paso = (
        domains.coffee.shops.shop_config(name) for name in ("cafe_de_barrio", "cafe_de_paso")
    )

    assert domains.coffee.shops.businesses() == ["cafe_de_barrio", "cafe_de_paso"]
    assert (barrio.tenant, paso.tenant) == ("coffee/cafe_de_barrio", "coffee/cafe_de_paso")
    assert [m.name for m in barrio.models] == [m.name for m in paso.models]
    assert [m.training.registered_model for m in (*barrio.models, *paso.models)] == [
        "cafe-de-barrio-hourly-demand",
        "cafe-de-barrio-order-minutes",
        "cafe-de-paso-hourly-demand",
        "cafe-de-paso-order-minutes",
    ]
    assert barrio.shop.zone_id != paso.shop.zone_id
    assert shop_adapter.config == barrio


def test_a_shops_file_describes_its_shop_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rogue.yaml").write_text("shop: {}\nparent: {domain: games}\n", encoding="utf-8")
    monkeypatch.setattr(domains.coffee.shops, "BUSINESSES", tmp_path)

    with pytest.raises(ValueError, match="describes its shop and nothing else"):
        domains.coffee.shops.shop_config("rogue")


def test_the_shop_reads_only_what_it_lists_of_coffee(shop_adapter: CoffeeShopAdapter) -> None:
    config = shop_adapter.config

    assert config.parent_tables == {
        "coffee.analysis.green_coffee_in_pesos",
        "coffee.clean.consumer_price_index",
        "coffee.analysis.price_outlook",
        "coffee.clean.coffee_shops",
        "coffee.features.zones_features",
        "coffee.predictions.zones_predictions",
        "coffee.clean.roaster_offers",
        "coffee.clean.census_zones",
    }
    as_json = config.model_dump(mode="json")
    undeclared = {**as_json["simulation"], "inflation": "coffee.clean.consumer_prices"}
    with pytest.raises(ValidationError, match="does not list"):
        CoffeeShopConfig.model_validate({**as_json, "simulation": undeclared})
    unlisted = {**as_json["studies"], "coffee_shops": "coffee.clean.coffee_shop_history"}
    with pytest.raises(ValidationError, match="The studies read"):
        CoffeeShopConfig.model_validate({**as_json, "studies": unlisted})
    unknown = {**as_json["simulation"], "sales_pattern": "pos_sales"}
    with pytest.raises(ValidationError, match="No source 'pos_sales'"):
        CoffeeShopConfig.model_validate({**as_json, "simulation": unknown})


def test_a_shop_config_refuses_what_it_could_not_simulate(
    shop_adapter: CoffeeShopAdapter,
) -> None:
    shop = shop_adapter.config.shop.model_dump(mode="json")

    with pytest.raises(ValidationError, match="does not buy"):
        ShopConfig.model_validate(
            {**shop, "menu": [{**shop["menu"][0], "recipe": {"oat_milk": 200}}]}
        )
    with pytest.raises(ValidationError, match="add up to 1"):
        ShopConfig.model_validate({**shop, "items_per_ticket": [0.5, 0.2]})
    with pytest.raises(ValidationError, match="also closes"):
        ShopConfig.model_validate({**shop, "closes": {"1": "21:00"}})


def test_the_anchors_recover_a_known_elasticity_and_the_hours() -> None:
    sales = pl.read_csv(io.BytesIO(vending_csv()), infer_schema_length=0)

    anchors = sales_anchors(sales, resamples=200)

    assert anchors.elasticity == pytest.approx(-1.0, abs=0.05)
    assert anchors.elasticity_low <= anchors.elasticity <= anchors.elasticity_high
    assert anchors.hour_shares == pytest.approx({8: 0.5, 15: 0.5}, abs=0.01)
    assert anchors.weekday_factors == pytest.approx(dict.fromkeys(range(1, 8), 1.0), abs=0.01)
    assert anchors.days == 56


def test_costs_move_from_the_month_the_shop_opened() -> None:
    green = pl.DataFrame(
        {
            "period": [date(2024, 12, 1), date(2025, 1, 1), date(2025, 2, 1), date(2025, 1, 1)],
            "indicator": ["other_milds", "other_milds", "other_milds", "robustas"],
            "mxn_per_kg": [100.0, 110.0, 132.0, 50.0],
        }
    )
    cpi = pl.DataFrame({"month": [date(2025, 1, 1), date(2025, 2, 1)], "index": [100.0, 101.0]})

    factors = cost_factors(green, cpi, "other_milds", date(2025, 1, 1))

    assert factors["green_coffee"][date(2025, 2, 1)] == pytest.approx(1.2)
    assert factors["green_coffee"][date(2024, 12, 1)] == pytest.approx(100 / 110)
    assert factors["inflation"][date(2025, 2, 1)] == pytest.approx(1.01)
    # A month not published yet keeps the last one's; before every month, nothing moved.
    assert factor_on(factors["green_coffee"], date(2025, 3, 15)) == pytest.approx(1.2)
    assert factor_on(factors["inflation"], date(2024, 6, 1)) == 1.0
    with pytest.raises(ValueError, match="No inflation month at or before"):
        cost_factors(green, cpi, "other_milds", date(2024, 12, 1))


def test_price_changes_compound_and_a_promotion_ends(shop_adapter: CoffeeShopAdapter) -> None:
    shop = shop_adapter.config.shop

    assert price_level(shop, date(2025, 2, 28)) == 1.0
    assert price_level(shop, date(2025, 3, 1)) == pytest.approx(1.06)
    assert price_level(shop, date(2025, 8, 15)) == pytest.approx(1.06 * 0.9)
    assert price_level(shop, date(2025, 9, 1)) == pytest.approx(1.06)
    assert price_level(shop, date(2026, 2, 1)) == pytest.approx(1.06 * 1.08)

    periods = _menu_history(shop)
    espresso = [row for row in periods if row["product"] == "espresso"]
    assert [row["valid_from"] for row in espresso] == [
        "2025-01-01",
        "2025-03-01",
        "2025-08-01",
        "2025-09-01",
        "2026-02-01",
    ]
    assert espresso[-1]["valid_to"] is None
    assert espresso[2] == {
        "product": "espresso",
        "category": "espresso",
        "price": 38.16,  # 40 x 1.06 x 0.9
        "valid_from": "2025-08-01",
        "valid_to": "2025-08-31",
    }


def test_the_simulation_is_the_same_every_time_and_keeps_the_shops_rules(
    shop_adapter: CoffeeShopAdapter,
) -> None:
    shop = two_weeks(shop_adapter.config.shop)
    moved = {**FLAT, "green_coffee": {date(2025, 1, 1): 2.0}}

    export = simulate(shop, ANCHORS, moved)

    assert export == simulate(shop, ANCHORS, moved)
    assert set(export) == set(TABLES)
    prices = {item.product: item.price_mxn for item in shop.menu}
    for row in export["pos_sales"]:
        sold = datetime.fromisoformat(row["sold_at"])
        weekday = sold.isoweekday()
        assert shop.opens[weekday] <= sold.time() < shop.closes[weekday]
        assert row["unit_price"] == prices[row["product"]]  # no change before March
    # A ticket is eaten in or taken away, and paid, once for all its lines.
    tickets = {(r["ticket"], r["channel"], r["payment"]) for r in export["pos_sales"]}
    assert len(tickets) == len(export["pos_orders"])
    assert {date.fromisoformat(row["date"]).isoweekday() for row in export["pos_purchases"]} == {1}
    # Half of what beans cost moves with green coffee, so twice the green price is 1.5x.
    beans = shop.ingredients["coffee_beans"]
    first = next(
        r
        for r in export["pos_purchases"]
        if r["ingredient"] == "coffee_beans" and r["date"] == "2025-01-06"
    )
    assert first["cost"] / first["quantity"] == pytest.approx(beans.cost_mxn * 1.5, rel=0.01)
    # Ten weekdays of two baristas and a helper, two Saturdays and two Sundays of two.
    assert len(export["pos_shifts"]) == 10 * 3 + 2 * 2 + 2 * 2


def test_a_shop_short_of_hands_sells_less_at_its_rush(shop_adapter: CoffeeShopAdapter) -> None:
    """Customers facing a long queue leave: with one barista the crowd is not served."""
    every_day = list(range(1, 8))
    alone = [Shift(role="barista", starts=time(7), ends=time(21), days=every_day)]
    crowd = {"tickets_per_day": 400.0}
    shop = shop_adapter.config.shop

    short = simulate(two_weeks(shop, shifts=alone, **crowd), ANCHORS, FLAT)
    staffed = simulate(two_weeks(shop, shifts=alone * 4, **crowd), ANCHORS, FLAT)

    assert len(short["pos_orders"]) < 0.8 * len(staffed["pos_orders"])


def test_extract_skips_out_loud_without_its_anchors(
    shop_adapter: CoffeeShopAdapter, tmp_path: Path
) -> None:
    home = tmp_path / "data" / shop_adapter.config.home
    with httpx.Client() as client:
        extraction = shop_adapter.extract(home, client)

    assert extraction.artifacts == {}
    assert set(extraction.skipped) == set(TABLES)
    assert all("run the coffee domain first" in why for why in extraction.skipped.values())


def test_the_export_is_simulated_from_what_coffee_lends_and_cleaned_by_the_core(
    shop_adapter: CoffeeShopAdapter, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    write_table(
        pl.DataFrame(
            {
                "period": [date(2024, 12, 1), date(2025, 1, 1)],
                "indicator": ["other_milds", "other_milds"],
                "mxn_per_kg": [120.0, 126.0],
            }
        ),
        data / "coffee" / "analysis" / "green_coffee_in_pesos",
        {},
    )
    write_table(
        pl.DataFrame({"month": [date(2025, 1, 1)], "index": [140.0]}),
        data / "coffee" / "clean" / "consumer_price_index",
        {},
    )
    shop_dir = data / "coffee" / "subdomains" / "cafe_de_barrio"
    store_payload("vending_sales", "coffee_sales.csv", vending_csv(), shop_dir / "raw", "https://x")
    config = shop_adapter.config
    adapter = CoffeeShopAdapter(config.model_copy(update={"shop": two_weeks(config.shop)}))
    at = datetime(2026, 10, 4, tzinfo=UTC)

    with httpx.Client() as client:
        first = adapter.extract(shop_dir, client, now=at)
        again = adapter.extract(shop_dir, client, now=at + timedelta(hours=1))

    assert set(first.artifacts) == set(TABLES)
    # The same anchors and config make the same bytes: nothing new is stored.
    assert {t: a.path for t, a in again.artifacts.items()} == {
        t: a.path for t, a in first.artifacts.items()
    }
    document = json.loads(first.artifacts["pos_menu"].path.read_text(encoding="utf-8"))
    assert document["anchors"]["elasticity"] == pytest.approx(-1.0, abs=0.05)

    built = build_clean(adapter, shop_dir, at)

    assert set(built) == set(CLEAN_SCHEMAS)


def test_the_export_cleans_into_the_shops_tables(shop_adapter: CoffeeShopAdapter) -> None:
    shop = two_weeks(shop_adapter.config.shop)
    adapter = CoffeeShopAdapter(shop_adapter.config.model_copy(update={"shop": shop}))
    raw = export_frames(simulate(shop, ANCHORS, FLAT))

    tables = {name: table.frame for name, table in adapter.clean(raw, {}).items()}

    for name, frame in tables.items():
        check_contract(CLEAN_SCHEMAS[name], frame)
    hours = tables["shop_hours"]
    # Ten weekdays of 14 open hours (07:30-21:00), two Saturdays of 13, two Sundays of 12,
    # whether or not they sold anything.
    assert hours.height == 10 * 14 + 2 * 13 + 2 * 12
    assert hours["revenue_mxn"].sum() == pytest.approx(tables["sales"]["line_total_mxn"].sum())
    assert hours["tickets"].sum() == tables["orders"].height
    monday = hours.filter(pl.col("date") == date(2025, 1, 6)).select("hour", "baristas").rows()
    on_shift = dict(monday)
    assert (on_shift[7], on_shift[8], on_shift[14], on_shift[16]) == (1, 2, 2, 1)
    assert hours["price_level"].unique().to_list() == [1.0]
    with pytest.raises(ValueError, match="No point-of-sale export"):
        adapter.clean({name: frame for name, frame in raw.items() if name != "pos_menu"}, {})


def test_an_export_with_no_sales_has_no_hours(shop_adapter: CoffeeShopAdapter) -> None:
    raw = export_frames(simulate(two_weeks(shop_adapter.config.shop), ANCHORS, FLAT))
    tables = clean_shop(raw, shop_adapter.config.shop)

    empty = shop_hours(
        tables["sales"].clear(),
        tables["orders"],
        tables["menu_prices"],
        tables["shifts"],
        shop_adapter.config.shop,
    )

    assert empty.is_empty()
    assert list(empty.columns) == list(CLEAN_SCHEMAS["shop_hours"].columns)
    assert to_frame({"rows": []}).is_empty()


def monday_eights(tickets: list[float]) -> pl.DataFrame:
    """Four Mondays at eight, the last weeks the shop's tables hold, at the latest prices."""
    days = [date(2026, 8, 10) + timedelta(days=7 * k) for k in range(len(tickets))]
    return pl.DataFrame(
        {
            "date": days,
            "weekday": [1] * len(days),
            "hour": [8] * len(days),
            "tickets": tickets,
            "price_level": [1.06 * 1.08] * len(days),
        }
    )


def test_an_hour_is_asked_at_the_menus_prices_or_moved(shop_adapter: CoffeeShopAdapter) -> None:
    menu = clean_menu(to_frame({"rows": _menu_history(shop_adapter.config.shop)}))
    context = {"menu_prices": menu, "shop_hours": monday_eights([10.0, 12.0, 14.0, 16.0])}
    asked = ShopHour(weekday=1, hour=8, price_change_pct=10)
    now = ShopHour(weekday=1, hour=8)
    batch = pl.DataFrame(
        {
            "hour_id": ["2025-03-03T08"],
            "date": [date(2025, 3, 3)],
            "hour": [8],
            "weekday": [1],
            "price_level": [9.9],
        }
    )

    online = add_hour_context(pl.DataFrame([asked.to_item(), now.to_item()]), context)
    scored = add_hour_context(batch, context)

    # The next Monday, at the menu in force, moved by the request's change.
    assert online["price_level"].to_list() == pytest.approx(
        [1.06 * 1.08 * 1.1, 1.06 * 1.08], rel=1e-3
    )
    assert online["weekday_hour"].to_list() == ["1-08", "1-08"]
    assert "price_change_pct" not in online.columns
    # Its history is the latest weeks the tables hold: last time 16, the four's mean 13.
    assert online["slot_last"].to_list() == [16.0, 16.0]
    assert online["slot_mean_4w"].to_list() == [13.0, 13.0]
    assert online["history_price_level"].to_list() == pytest.approx([1.06 * 1.08] * 2)
    # A batch hour takes the menu's level, the same function as a request's, and the
    # history before its day - none, before the tables' first Monday.
    assert scored["price_level"].to_list() == pytest.approx([1.06], rel=1e-3)
    assert scored["slot_last"].to_list() == [None]


def test_the_shops_model_is_answered_by_its_hooks(shop_adapter: CoffeeShopAdapter) -> None:
    model = shop_adapter.config.model_named("hourly_demand")

    assert shop_adapter.context_tables("hourly_demand") == ("menu_prices", "shop_hours")
    assert shop_adapter.request_model("hourly_demand") is ShopHour
    assert ShopHour.model_validate(model.example).to_item()["weekday"] == 1
    assert shop_adapter.raw_contracts().keys() == {"vending_sales", *TABLES}
    assert set(shop_adapter.json_readers()) == set(TABLES)
    assert shop_adapter.credentials() == {}
    assert shop_adapter.file_readers() == {}
    assert shop_adapter.studies({}) == {}
    assert shop_adapter.figures({}) == {}
    items = pl.DataFrame({"date": [date(2025, 1, 6)], "hour": [9], "weekday": [1]})
    menu = clean_menu(to_frame({"rows": _menu_history(shop_adapter.config.shop)}))
    enriched = shop_adapter.enrich(
        "hourly_demand", items, {"menu_prices": menu, "shop_hours": monday_eights([10.0])}
    )
    assert enriched["weekday_hour"].to_list() == ["1-09"]
    with pytest.raises(ValueError, match="no code for model 'margins'"):
        shop_adapter.context_tables("margins")


def test_the_data_dictionary_documents_every_table_and_column() -> None:
    dictionary = (domain_dir("coffee/cafe_de_barrio") / "data_dictionary.md").read_text(
        encoding="utf-8"
    )

    for table, schema in CLEAN_SCHEMAS.items():
        assert f"## `clean.{table}`" in dictionary
        assert [c for c in schema.columns if f"`{c}`" not in dictionary] == []
    for name in ("hourly_demand_features", "hourly_demand_predictions", "weekday_hour"):
        assert f"{name}`" in dictionary
