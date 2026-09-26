"""The international price of green coffee: the ICO's page read and checked, the World
Bank's months converted, and the two stacked into one table.

The ICO's page is written here line by line, in the shape pypdf reads the real one.
"""

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from pandera.errors import SchemaErrors

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.prices import clean_price_indicators, reconcile_prices
from domains.coffee.schemas import ICO_PRICES, PRICE_INDICATORS
from domains.coffee.sources.ico import parse_indicator_prices, read_indicator_prices
from mlops_core.contracts import check_contract
from mlops_core.data.clean import build_clean
from mlops_core.data.extract import ingestions
from mlops_core.data.validate import validate_raw
from mlops_core.storage import read_table
from tests.conftest import ICO_PAGE
from tests.files import pdf

T0 = datetime(2026, 9, 3, 18, tzinfo=UTC)
T1 = datetime(2026, 9, 4, 18, tzinfo=UTC)


def page(*changes: tuple[str, str]) -> str:
    text = "\n".join(ICO_PAGE)
    for old, new in changes:
        text = text.replace(old, new)
    return text


# --- The ICO's page ---------------------------------------------------------------------


def test_the_page_is_read_to_its_days_and_checked_against_its_own_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "I-CIP.pdf"
    path.write_bytes(pdf(ICO_PAGE))

    days = read_indicator_prices(path)

    assert days["date"].to_list() == ["2026-09-01", "2026-09-02", "2026-09-03"]  # 4-Sep blank
    assert days.row(0) == ("2026-09-01", "279.85", "380.60", "352.86", "315.48", "172.99")
    assert check_contract(ICO_PRICES, days)["other_milds"].dtype == pl.Float64


def test_a_misread_page_is_refused_rather_than_stored() -> None:
    """One figure read wrong and the days no longer average to the page's own Average."""
    with pytest.raises(ValueError, match="other_milds: the days read give Average"):
        parse_indicator_prices(page(("340.80", "34.80")))
    with pytest.raises(ValueError, match="i_cip: the days read give High"):
        parse_indicator_prices(page(("High 279.85", "High 289.85")))
    with pytest.raises(ValueError, match="no \\['Low'\\] row"):
        parse_indicator_prices(page(("Low 268.38 366.51 338.76 300.83 166.66", "")))


def test_a_page_laid_out_otherwise_is_refused() -> None:
    with pytest.raises(ValueError, match="The columns changed"):
        parse_indicator_prices(page((" Naturals Robustas", " Naturals Robustas Excelsas")))
    with pytest.raises(ValueError, match="A Aug row on the September page"):
        parse_indicator_prices(page(("3-Sep", "3-Aug")))
    with pytest.raises(ValueError, match="Not the ICO's indicator price page"):
        parse_indicator_prices(page(("In US cents/lb", "In euros/kg")))


def test_a_month_with_no_prices_yet_is_an_empty_reading() -> None:
    """The first morning of a month: every day blank, and nothing to check."""
    blank = "\n".join(
        line.split(" ")[0] + "     " if line[:1].isdigit() else line
        for line in ICO_PAGE
        if not line.startswith(("Average", "High", "Low"))
    )

    days = parse_indicator_prices(blank)

    assert days.is_empty() and days.columns[0] == "date"
    with pytest.raises(ValueError, match="No row of days on the page"):
        parse_indicator_prices("\n".join(line for line in ICO_PAGE if not line[:1].isdigit()))


# --- One table from both publishers -----------------------------------------------------


def daily(*reads: tuple[datetime, float]) -> pl.DataFrame:
    """The ICO's first of September as read at each time, with its value then."""
    return pl.DataFrame(
        {
            "date": ["2026-09-01"] * len(reads),
            **{name: [100.0] * len(reads) for name in ("i_cip", "colombian_milds",
                                                        "brazilian_naturals", "robustas")},
            "other_milds": [value for _, value in reads],
            "ingested_at": [at for at, _ in reads],
        }
    )  # fmt: skip


def monthly(arabica: float) -> pl.DataFrame:
    return pl.DataFrame(
        {"column_1": ["2026M09"], "Coffee, Arabica": [arabica], "Coffee, Robusta": [4.0]}
    )


def test_a_day_read_twice_keeps_its_latest_reading_and_months_are_converted() -> None:
    table = clean_price_indicators(daily((T1, 352.0), (T0, 350.0)), monthly(7.0), T1)

    check_contract(PRICE_INDICATORS, table)
    day = table.filter(pl.col("indicator") == "other_milds", pl.col("frequency") == "daily")
    assert (day["usd_cents_per_lb"].item(), day["read_at"].item()) == (352.0, T1)
    month = table.filter(pl.col("indicator") == "other_milds", pl.col("frequency") == "monthly")
    assert month["period"].item() == date(2026, 9, 1)
    assert month["usd_cents_per_lb"].item() == pytest.approx(7.0 * 100 / 2.20462262)
    assert month["source"].item() == "world_bank"


def test_the_two_publishers_are_compared_where_both_have_the_month(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = clean_price_indicators(daily((T0, 317.0)), monthly(7.0), T1)

    gaps = reconcile_prices(table)

    assert gaps[("2026-09", "other_milds")] == pytest.approx(7.0 * 45.359237 - 317.0)
    assert ("2026-09", "robustas") in gaps
    with caplog.at_level(logging.INFO):
        reconcile_prices(table.filter(pl.col("frequency") == "daily"))
    assert "no month is in both" in caplog.text


def test_the_clean_layer_stacks_every_download_of_the_page(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    """Two downloads of September, the second with a day more: the history is both."""
    first = ingestions(raw_dir, "ico_prices")[0]
    later = raw_dir / "ico_prices" / "ingested_at=20991231T000000000000Z"
    later.mkdir()
    fourth = "4-Sep 266.02 362.00 334.02 297.02 165.00"
    lines = [fourth if line == "4-Sep     " else line for line in ICO_PAGE]
    lines = [line if not line.startswith(("Average", "High", "Low")) else "" for line in lines]
    lines += [
        "Average 271.18 369.41 341.61 304.10 168.33",
        "High 279.85 380.60 352.86 315.48 172.99",
        "Low 266.02 362.00 334.02 297.02 165.00",
    ]
    (later / first.manifest.filename).write_bytes(pdf(lines))
    manifest = first.manifest.model_copy(update={"ingested_at": datetime(2099, 12, 31, tzinfo=UTC)})
    (later / "manifest.json").write_text(manifest.model_dump_json())

    validated = validate_raw(coffee_adapter, raw_dir)
    build_clean(coffee_adapter, raw_dir.parent)
    prices = read_table(raw_dir.parent / "clean" / "price_indicators")

    assert validated["ico_prices"].frame.height == 3 + 4
    assert validated["ico_prices"].lineage == f"{later.name} and 1 earlier"
    days = prices.filter(pl.col("source") == "ico", pl.col("indicator") == "i_cip")
    assert days["period"].to_list() == [date(2026, 9, d) for d in (1, 2, 3, 4)]


def test_a_price_table_with_a_repeated_period_breaks_its_contract() -> None:
    table = clean_price_indicators(daily((T0, 350.0)), monthly(7.0), T1)

    with pytest.raises(SchemaErrors):
        check_contract(PRICE_INDICATORS, pl.concat([table, table.head(1)]))
