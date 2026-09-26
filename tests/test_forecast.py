"""The green_price model's enrich: what a month may know, and only that.

Prices are written here month by month; a test reads its own series.
"""

from datetime import date

import polars as pl
import pytest

from domains.coffee.forecast import add_price_history
from domains.coffee.request import PriceMonth


def prices(series: dict[str, list[float]], start: date = date(2024, 1, 1)) -> pl.DataFrame:
    """Monthly rows of `price_indicators`, one series per indicator from `start`, and a
    daily row that the model must ignore."""
    rows = [
        {
            "period": date(
                start.year + (start.month - 1 + n) // 12, (start.month - 1 + n) % 12 + 1, 1
            ),
            "frequency": "monthly",
            "indicator": indicator,
            "usd_cents_per_lb": value,
        }
        for indicator, values in series.items()
        for n, value in enumerate(values)
    ]
    daily = {"period": date(2024, 2, 15), "frequency": "daily", "indicator": "i_cip",
             "usd_cents_per_lb": 999.0}  # fmt: skip
    return pl.DataFrame([*rows, daily])


def month(frame: pl.DataFrame, month_id: str) -> dict[str, object]:
    return frame.filter(pl.col("month_id") == month_id).row(0, named=True)


def test_a_month_knows_the_month_before_it_and_is_judged_on_its_own_change() -> None:
    table = prices({"other_milds": [100.0, 110.0, 99.0], "robustas": [50.0, 55.0, 44.0]})

    features = add_price_history(table, table)
    march = month(features, "other_milds-2024-03")

    assert (march["price_last"], march["change_last"]) == (110.0, pytest.approx(10.0))
    assert march["change_2_back"] is None  # January has no month before it
    assert march["change_pct"] == pytest.approx(-10.0)  # its own change: 110 -> 99
    assert march["arabica_robusta_ratio"] == pytest.approx(2.0)  # February's, not March's
    assert (march["decade"], march["calendar_month"]) == ("2020s", "03")
    # January has nothing before it to change from: no item. The daily row is not one.
    assert sorted(features["month_id"]) == [
        "other_milds-2024-02", "other_milds-2024-03", "robustas-2024-02", "robustas-2024-03"
    ]  # fmt: skip


def test_a_gap_leaves_a_feature_empty_instead_of_borrowing_another_month() -> None:
    """With March missing, April's "month before" is not February."""
    table = prices({"other_milds": [100.0, 110.0, 99.0, 90.0]}).filter(
        pl.col("period") != date(2024, 3, 1)
    )

    features = add_price_history(table, table)

    assert "other_milds-2024-04" not in features["month_id"].to_list()  # nothing to change from
    assert month(features, "other_milds-2024-02")["arabica_robusta_ratio"] is None  # no robustas


def test_rolling_figures_need_a_full_window() -> None:
    series = [100.0 + n for n in range(14)]
    table = prices({"other_milds": series})

    features = add_price_history(table, table)

    assert month(features, "other_milds-2024-12")["gap_to_mean_12m"] is None  # 11 months known
    thirteenth = month(features, "other_milds-2025-01")
    assert thirteenth["gap_to_mean_12m"] == pytest.approx((111 / (sum(series[:12]) / 12) - 1) * 100)
    assert month(features, "other_milds-2024-07")["volatility_6m"] is None  # 5 changes known
    assert month(features, "other_milds-2024-08")["volatility_6m"] is not None
    assert thirteenth["change_12m"] is None  # needs the month twelve before the one before
    assert month(features, "other_milds-2025-02")["change_12m"] == pytest.approx(
        (112 / 100 - 1) * 100
    )


def test_a_month_asked_online_gets_what_the_batch_row_for_it_got() -> None:
    """The API's path and the feature table's cannot compute a feature differently."""
    table = prices({"other_milds": [100.0, 104.0, 99.0, 103.0], "robustas": [50.0, 51, 49, 48]})
    batch = month(add_price_history(table, table), "other_milds-2024-04")
    asked = PriceMonth(indicator="other_milds", month=date(2024, 4, 22))

    online = add_price_history(pl.DataFrame([asked.to_item()]), table).row(0, named=True)

    assert online["change_pct"] is None  # its own price is what is being asked
    features = [c for c in batch if c not in ("usd_cents_per_lb", "change_pct")]
    assert {c: online[c] for c in features} == {c: batch[c] for c in features}
