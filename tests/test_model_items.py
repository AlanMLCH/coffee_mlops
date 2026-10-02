"""The six models of v1.1.1, as the domain describes their items: what each knows, what
it predicts, and that a request reaches the same row the batch builds."""

from datetime import date

import polars as pl
import pytest

from domains.coffee.auction import add_lot_market, market_by_year
from domains.coffee.buyers import add_household_traits
from domains.coffee.outlook import HORIZONS, add_price_outlook, price_outlook
from domains.coffee.request import AuctionLot, Household, Place, PriceOutlook, ShelfItem, Zone
from domains.coffee.shelf import add_shelf_context
from domains.coffee.shop_kind import NAME_WORDS, add_place_traits, name_words
from domains.coffee.zone_profile import add_zone_profile, zone_profile


def zones_context() -> dict[str, pl.DataFrame]:
    zones = pl.DataFrame(
        {
            "zone_id": ["0900200010010", "0900200010025", "0901500010010"],
            "borough_id": ["09002", "09002", "09015"],
            "borough": ["Azcapotzalco", "Azcapotzalco", "Cuauhtémoc"],
            "area_km2": [0.5, 0.25, 1.0],
            "population": [1000, 0, 4000],
            "dwellings": [400, 0, 1600],
            "schooling_years": [10.0, None, 14.0],
            "economically_active": [500, 0, 2400],
            "people_65_plus": [100, 0, 400],
            "dwellings_with_internet": [200, 0, 1200],
            "dwellings_with_car": [100, 0, 800],
            "dwellings_with_computer": [100, 0, 1000],
        }
    )
    shops = pl.DataFrame(
        {
            "shop_id": ["a", "b", "c", "d", "e", "f"],
            "source": ["denue", "denue", "denue", "denue", "osm", "denue"],
            "name": ["CAFE UNO", "JUGOS", "LA ESQUINA", "CAFE DOS", "Café", "CREPAS Y CAFE"],
            "kind": ["coffee", "juice", "unclassified", "coffee", "coffee", "coffee"],
            "zone_id": [
                "0901500010010",
                "0901500010010",
                "0901500010010",
                "0900200010010",
                "0901500010010",
                None,
            ],
            "borough_id": ["09015", "09015", "09015", "09002", "09015", "09015"],
            "employees_band": ["0 a 5 personas"] * 6,
            "listed_since": [date(2010, 7, 1)] * 6,
            "matched_shop_id": [None] * 6,
        }
    )
    stations = pl.DataFrame(
        {
            "station_id": ["m1", "b1"],
            "system": ["metro", "metrobus"],
            "zone_id": ["0901500010010", "0901500010010"],
        }
    )
    boroughs = pl.DataFrame(
        {
            "borough_id": ["09002", "09015"],
            "population": [400_000, 500_000],
            "jobs_estimate": [200_000.0, 1_000_000.0],
        }
    )
    return {"census_zones": zones, "coffee_shops": shops, "transit_stations": stations,
            "boroughs": boroughs}  # fmt: skip


def test_a_zone_is_counted_from_the_registers_and_described_by_the_census() -> None:
    context = zones_context()

    profile = zone_profile(*context.values()).sort("zone_id")
    centre = profile.row(2, named=True)

    assert profile["coffee_shops"].to_list() == [1, 0, 1]  # OSM's coffee shop is not DENUE's
    assert centre["other_places"] == 1  # the juice bar; the unclassified one is not counted
    assert (centre["metro_stations"], centre["metrobus_stations"]) == (1, 1)
    assert centre["borough_jobs_per_resident"] == 2.0
    assert centre["internet_pct"] == 75.0 and centre["active_pct"] == 60.0
    empty = profile.row(1, named=True)
    assert empty["internet_pct"] is None and empty["aged_65_plus_pct"] is None  # no one lives there


def test_a_zone_asked_about_online_is_the_zone_the_batch_scored() -> None:
    context = zones_context()

    batch = add_zone_profile(context["census_zones"], context)
    online = add_zone_profile(pl.DataFrame([Zone(zone_id="0901500010010").to_item()]), context)

    assert online.row(0) == batch.filter(pl.col("zone_id") == "0901500010010").row(0)
    assert batch["census"].unique().to_list() == ["2020"]


def prices(months: int = 30) -> pl.DataFrame:
    periods = pl.Series([date(2024 + (m // 12), m % 12 + 1, 1) for m in range(months)])
    return pl.DataFrame(
        {
            "period": list(periods) * 2,
            "frequency": ["monthly"] * (2 * months),
            "indicator": ["other_milds"] * months + ["robustas"] * months,
            "usd_cents_per_lb": [200.0 + 5 * m for m in range(months)]
            + [100.0 + 2 * m for m in range(months)],
        }
    )


def test_a_month_is_asked_at_each_horizon_and_the_months_ahead_are_scored() -> None:
    table = prices()

    outlook = add_price_outlook(table, {"price_indicators": table})

    known = outlook.filter(pl.col("change_pct").is_not_null())
    ahead = outlook.filter(pl.col("usd_cents_per_lb").is_null())
    assert sorted(ahead["outlook_id"].to_list()) == [
        f"{indicator}-{month}-{h}m"
        for indicator in ("other_milds", "robustas")
        for h, month in zip(HORIZONS, ("2026-09", "2026-12", "2027-06"), strict=True)
    ]
    row = known.filter(pl.col("outlook_id") == "other_milds-2025-06-12m").row(0, named=True)
    # From June 2024 (225) to June 2025 (285), knowing only June 2024.
    assert row["price_last"] == 225.0
    assert row["change_pct"] == pytest.approx((285 / 225 - 1) * 100)
    assert row["horizon_months"] == 12.0 and row["year"] == "2025"


def test_a_request_for_a_horizon_is_the_batch_row_ahead() -> None:
    table = prices()
    context = {"price_indicators": table}

    batch = add_price_outlook(table, context)
    online = add_price_outlook(
        pl.DataFrame([PriceOutlook(indicator="robustas", months_ahead=6).to_item()]), context
    )

    expected = batch.filter(pl.col("outlook_id") == "robustas-2026-12-6m")
    assert online.select(expected.columns).row(0) == expected.row(0)


def readings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2025, 1, 10), date(2025, 1, 10), date(2026, 7, 2)],
            "product": ["instant"] * 3,
            "brand": ["Nescafé. Clásico"] * 3,
            "presentation": ["Frasco 200 Gr."] * 3,
            "grams": [200.0] * 3,
            "sweetened": [False] * 3,
            "decaf": [False] * 3,
            "price_mxn": [100.0, 110.0, 120.0],
            "chain": ["Wal-mart"] * 3,
            "store_type": ["Supermercado / Tienda de Autoservicio"] * 3,
            "store": ["s1"] * 3,
            "state": ["Ciudad de México"] * 3,
        }
    )


def test_a_shelf_price_is_said_in_pesos_of_the_latest_month() -> None:
    index = pl.DataFrame({"month": [date(2025, 1, 1), date(2026, 7, 1)], "index": [100.0, 110.0]})

    shelf = add_shelf_context(readings(), {"consumer_price_index": index}).sort("date")

    # The jar recorded twice the same day is one reading, at its mean price.
    assert shelf["price_mxn"].to_list() == [105.0, 120.0]
    assert shelf["price_today_mxn"].to_list() == pytest.approx([115.5, 120.0])
    assert shelf["product_name"][0] == "nescafe clasico 200 g"
    assert shelf["brand"][0] == "nescafe clasico" and shelf["chain"][0] == "wal mart"


def test_a_shelf_request_reaches_the_model_as_a_reading_does() -> None:
    index = pl.DataFrame({"month": [date(2026, 7, 1)], "index": [110.0]})
    request = ShelfItem(
        brand="Nescafé Clásico",
        grams=200,
        product="instant",
        chain="Wal-mart",
        store_type="Supermercado / Tienda de Autoservicio",
        state="Ciudad de México",
        observed_on=date(2026, 7, 2),
    )
    context = {"consumer_price_index": index}

    online = add_shelf_context(pl.DataFrame([request.to_item()]), context)
    batch = add_shelf_context(readings().tail(1), context)

    features = ["product_name", "brand", "product", "chain", "store_type", "state", "grams",
                "sweetened", "decaf"]  # fmt: skip
    assert online.select(features).row(0) == batch.select(features).row(0)
    assert online["price_today_mxn"][0] is None


def test_without_the_index_a_shelf_price_stays_in_its_own_pesos() -> None:
    empty = pl.DataFrame(schema={"month": pl.Date, "index": pl.Float64})

    shelf = add_shelf_context(readings(), {"consumer_price_index": empty})

    assert shelf["price_today_mxn"].to_list() == shelf["price_mxn"].to_list()


def test_a_place_is_known_by_its_other_words_and_learned_from_where_a_rule_spoke() -> None:
    context = zones_context()

    places = add_place_traits(context["coffee_shops"], context).sort("shop_id")

    assert places["shop_id"].to_list() == ["a", "b", "c", "d", "f"]  # DENUE's only
    assert places["is_coffee"].to_list() == [1.0, 0.0, None, 1.0, 1.0]
    crepes = places.row(4, named=True)
    assert crepes["name_crepas"] == 1.0 and crepes["block"] == "none-f"  # no zone: alone
    assert places.row(2, named=True)["metro_stations"] == 1
    assert name_words("Crepería Ñandú 2") == ["creperia", "nandu", "2"]
    assert len(set(NAME_WORDS)) == 30


def test_a_place_asked_about_online_is_described_as_the_batch_describes_it() -> None:
    context = zones_context()
    request = Place(name="LA ESQUINA", employees_band="0 a 5 personas", zone_id="0901500010010")

    online = add_place_traits(pl.DataFrame([request.to_item()]), context)
    batch = add_place_traits(context["coffee_shops"], context).filter(pl.col("shop_id") == "c")

    same = ["borough_id", "block", "name_word_count", "people_per_km2", "metro_stations",
            *[f"name_{word}" for word in NAME_WORDS]]  # fmt: skip
    assert online.select(same).row(0) == batch.select(same).row(0)


def lots() -> pl.DataFrame:
    """Three lots sold in 2024 and one that did not sell in 2025."""
    return pl.DataFrame(
        {
            "year": [2024, 2024, 2024, 2025],
            "lot_id": ["2024-001", "2024-002", "2024-003", "2025-001"],
            "score": [86.5, 88.0, 89.0, 90.2],
            "national_winner": [True, False, False, False],
            "state": ["Veracruz", "Oaxaca", "Puebla", "Chiapas"],
            "processing_method": ["washed", "honey", "washed", "natural"],
            "varieties": [["typica"], ["bourbon"], None, ["gesha", "bourbon"]],
            "weight_kg": [300.0, 200.0, 100.0, 60.0],
            "price_usd_per_lb": [5.0, 10.0, 20.0, None],
        }
    )


def test_a_lot_is_priced_against_the_other_lots_of_its_auction() -> None:
    context = {"price_indicators": prices(30), "cup_of_excellence": lots()}

    priced = add_lot_market(lots(), context).sort("lot_id")

    first, unsold = priced.row(0, named=True), priced.row(3, named=True)
    # Against the median of the other two, never its own price: (10 + 20) / 2.
    assert first["auction_median_usd_per_lb"] == 15.0
    assert first["premium_pct"] == pytest.approx((5.0 / 15.0 - 1) * 100)
    assert (first["score_band"], unsold["score_band"]) == ("under 87", "90 and over")
    assert (unsold["gesha"], unsold["varieties_n"]) == (1.0, 2.0)
    assert priced.row(2, named=True)["varieties_n"] == 0.0
    # No lot sold in 2025: the latest auction's median, 2024's; and no target.
    assert unsold["auction_median_usd_per_lb"] == 10.0
    assert unsold["premium_pct"] is None
    market_2024 = market_by_year(prices(30)).filter(pl.col("year") == 2024)["market_usd_per_lb"]
    assert first["market_usd_per_lb"] == market_2024[0]


def test_a_lot_asked_about_online_meets_the_same_auction() -> None:
    context = {"price_indicators": prices(30), "cup_of_excellence": lots()}
    request = AuctionLot(score=89.5, state="Veracruz", processing_method="Washed",
                         varieties=["Gesha"], year=2024)  # fmt: skip

    online = add_lot_market(pl.DataFrame([request.to_item()]), context)

    row = online.row(0, named=True)
    assert row["gesha"] == 1.0 and row["processing_method"] == "washed"
    assert row["auction_median_usd_per_lb"] == 10.0  # all three of 2024's lots are others
    assert row["market_usd_per_lb"] == pytest.approx((200 + 5 * 5.5) / 100)


def test_a_household_bought_coffee_if_it_paid_for_any() -> None:
    households = pl.DataFrame(
        {
            "household_id": ["h1", "h2", "h3"],
            "year": [2024] * 3,
            "state": ["Ciudad de México"] * 3,
            "members": [4, 1, 2],
            "income_quarter_mxn": [60_000.0, 9_000.0, 0.0],
            "instant_quarter_mxn": [0.0, 120.0, 0.0],
            "ground_quarter_mxn": [0.0, 0.0, 0.0],
            "prepared_quarter_mxn": [0.0, 0.0, 0.0],
            "own_harvest_quarter_mxn": [0.0, 0.0, 30.0],
        }
    )

    traits = add_household_traits(households)

    assert traits["buys_coffee"].to_list() == [0.0, 1.0, 0.0]
    assert traits["income_per_member_mxn"].to_list() == [15_000.0, 9_000.0, 0.0]
    assert traits["grows_coffee"].to_list() == [0.0, 0.0, 1.0]
    online = add_household_traits(
        pl.DataFrame([Household(state="Veracruz", members=3, income_month_mxn=10_000).to_item()])
    )
    assert online["buys_coffee"][0] is None
    assert online["income_quarter_mxn"][0] == 30_000.0


def test_the_outlook_is_each_horizons_historical_range_set_on_the_last_price() -> None:
    outlook = price_outlook(prices(30)).sort("indicator", "horizon_months")

    row = outlook.row(2, named=True)  # other milds, 12 months
    # Prices rise 5 cents a month from 200: every 12-month change is +60 cents.
    assert (row["from_month"], row["to_month"]) == (date(2026, 6, 1), date(2027, 6, 1))
    assert row["price_now"] == 345.0 and row["months"] == 18
    assert row["change_low_pct"] <= row["change_median_pct"] <= row["change_high_pct"]
    assert row["price_high"] == pytest.approx(345.0 * (1 + row["change_high_pct"] / 100))
    assert price_outlook(prices(30).clear()).is_empty()
    short = price_outlook(prices(8))  # eight months: nothing to say twelve months ahead
    assert sorted(set(short["horizon_months"])) == [3, 6]
