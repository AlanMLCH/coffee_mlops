"""The three studies of v1.1.1: how far a move in green coffee travels, what kinds of
coffee municipalities there are, and how the chains price."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from domains.coffee.segments import (
    MIN_READINGS,
    MIN_STORES,
    MIN_YEARS,
    chain_strategies,
    municipality_types,
    price_transmission,
)


def months(n: int, start: date = date(2024, 1, 1)) -> list[date]:
    return [date(start.year + (start.month - 1 + m) // 12, (start.month - 1 + m) % 12 + 1, 1)
            for m in range(n)]  # fmt: skip


def test_a_shelf_that_follows_green_coffee_a_month_later_passes_all_of_it_on() -> None:
    rng = np.random.default_rng(1)
    n = 40
    green_moves = rng.normal(0, 0.05, n)
    green_level = 100 * np.exp(np.cumsum(green_moves))
    # The shelf moves half as much as green coffee did the month before.
    shelf_level = 200 * np.exp(np.concatenate([[0.0], np.cumsum(0.5 * green_moves[:-1])]))
    periods = months(n)
    green = pl.DataFrame(
        {
            "period": periods * 2,
            "indicator": ["other_milds"] * n + ["robustas"] * n,
            "mxn_per_kg": [*green_level, *green_level],
        }
    )
    shelf = pl.DataFrame(
        {
            "date": periods * 2,
            "product": ["ground"] * n + ["instant"] * n,
            "sweetened": [False] * (2 * n),
            "decaf": [False] * (2 * n),
            "price_mxn_per_kg": [*shelf_level, *shelf_level],
        }
    )
    production = pl.DataFrame(schema={"year": pl.Int64, "value_mxn": pl.Float64,
                                      "production_t": pl.Float64})  # fmt: skip

    table = price_transmission(green, shelf, production)

    ground = table.filter(pl.col("step") == "ground coffee on a shelf").sort("lag")
    assert ground["pass_through"][0] == pytest.approx(0.0, abs=1e-6)  # nothing the same month
    assert ground["pass_through"][1] == pytest.approx(0.5)  # half of it, a month later
    assert (ground["ci_low"] <= ground["pass_through"]).all()
    assert table.filter(pl.col("frequency") == "annual").is_empty()  # no farm gate given


def test_too_short_a_series_says_nothing_rather_than_something() -> None:
    periods = months(5)
    green = pl.DataFrame(
        {"period": periods, "indicator": ["other_milds"] * 5, "mxn_per_kg": [1.0, 2, 3, 2, 1]}
    )
    shelf = pl.DataFrame(
        {
            "date": periods,
            "product": ["ground"] * 5,
            "sweetened": [False] * 5,
            "decaf": [False] * 5,
            "price_mxn_per_kg": [5.0, 6, 7, 6, 5],
        }
    )
    production = pl.DataFrame(
        {"year": [2024, 2025], "value_mxn": [10.0, 12.0], "production_t": [1.0, 1.0]}
    )

    assert price_transmission(green, shelf, production).is_empty()


def production(municipalities: int = 30, years: int = MIN_YEARS) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for m in range(municipalities):
        size = 100.0 * (1 + m % 5)
        for year in range(2010, 2010 + years):
            harvested = size * (1.02 if m % 2 else 0.98) ** (year - 2010)
            yield_ = 1.0 + (m % 3) * 0.5 + rng.normal(0, 0.05)
            rows.append(
                {"year": year, "state_id": f"{m % 2:02d}", "state": f"S{m % 2}",
                 "municipality_id": f"{m:05d}", "municipality": f"M{m}",
                 "planted_ha": harvested * 1.05, "harvested_ha": harvested, "lost_ha": 0.0,
                 "production_t": harvested * yield_, "value_mxn": harvested * yield_ * 5000,
                 "yield_t_per_ha": yield_, "rural_price_mxn_per_t": 5000.0 * (1 + (m % 4) / 10)}
            )  # fmt: skip
    return pl.DataFrame(rows)


def test_municipalities_are_grouped_by_their_ranked_traits_and_each_type_is_named() -> None:
    municipalities, types = municipality_types(production())

    assert municipalities.height == 30
    assert types["municipalities"].sum() == 30
    assert types["k"].n_unique() == 1 and 2 <= types["k"][0] <= 6
    assert set(municipalities["type"]) == set(types["type"])
    assert all(isinstance(name, str) and name for name in types["type"])


def test_too_few_municipalities_with_a_history_are_not_grouped() -> None:
    municipalities, types = municipality_types(production(municipalities=4, years=MIN_YEARS))

    assert municipalities.is_empty() and types.is_empty()
    assert "type" in municipalities.columns and "silhouette" in types.columns


def shelf_readings() -> pl.DataFrame:
    """Two chains over two months: one asks 10% more than the other for the same jar."""
    rows = []
    for chain, markup in (("dear", 1.1), ("cheap", 1.0)):
        for store in range(MIN_STORES):
            for day in range(MIN_READINGS // MIN_STORES + 1):
                rows.append(
                    {"chain": chain, "store_type": "Supermercado", "store": f"{chain}{store}",
                     "brand": "B", "presentation": "Frasco 200 Gr.",
                     "date": date(2025, 1 + day % 2, 1 + day % 28),
                     # One store in ten readings cuts its price by a fifth.
                     "price_mxn": 100.0 * markup * (0.8 if day % 10 == 0 else 1.0)}
                )  # fmt: skip
    return pl.DataFrame(rows)


def test_a_chain_is_premium_or_discount_only_when_its_interval_says_so() -> None:
    chains = chain_strategies(shelf_readings())

    dear = chains.filter(pl.col("chain") == "dear").row(0, named=True)
    cheap = chains.filter(pl.col("chain") == "cheap").row(0, named=True)
    assert dear["price_index"] > 1 > cheap["price_index"]
    assert dear["strategy"].startswith("premium") and cheap["strategy"].startswith("discount")
    assert dear["cut_pct"] == pytest.approx(10, abs=1)


def test_a_chain_seen_in_few_stores_is_not_judged() -> None:
    few = shelf_readings().filter(pl.col("store").is_in(["dear0", "cheap0"]))

    assert chain_strategies(few).is_empty()
