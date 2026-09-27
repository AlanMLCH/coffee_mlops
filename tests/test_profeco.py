"""PROFECO's fortnights: every file its own character set, columns and date format.

The archive the tests read is written, not recorded (`tests.conftest.QQP_FORTNIGHTS`):
the real one is 195 MB of a third party's data. Each of its three fortnights is one of
the ways the real files were found to differ, and a test here breaks each of them.
"""

import logging
import re
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from domains.coffee.sources.profeco import COLUMNS, fortnight, read_shelf_prices, record_date
from tests.conftest import (
    GROUND,
    INSTANT,
    POLANCO,
    QQP_COLUMNS,
    QQP_FORTNIGHTS,
    qqp_archive,
    shelf,
)

PRODUCTS = [INSTANT, GROUND]


def archive(tmp_path: Path, fortnights: dict[str, tuple[str, list[str], list[list[str]]]]) -> Path:
    path = tmp_path / "QQP_2026.zip"
    path.write_bytes(qqp_archive(fortnights))
    return path


def one_fortnight(
    member: str, *rows: list[str], encoding: str = "utf-8-sig"
) -> dict[str, tuple[str, list[str], list[list[str]]]]:
    return {member: (encoding, QQP_COLUMNS, list(rows))}


def test_every_fortnight_is_read_whatever_its_character_set(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    prices = read_shelf_prices(archive(tmp_path, QQP_FORTNIGHTS), "QQP_2026", PRODUCTS)

    assert prices.columns == [*COLUMNS, "file"]
    assert prices.height == 10  # coffee only: no coffee maker, no milk, no capsules
    assert set(prices["producto"]) == set(PRODUCTS)
    # May's file is cp1252 without a mark; its accents come through as accents.
    may = prices.filter(pl.col("file") == "QQP_2026/05-2026_Q1.csv")
    assert "Nescafé. Clásico" in may["marca"].to_list()
    assert "QQP_2026/05-2026_Q1.csv is not UTF-8: read as cp1252" in caplog.text
    # June's undocumented columns are left out, and said to be.
    assert "leaving out undocumented columns ['folio', 'cv_producto', 'cv_marca']" in caplog.text


def test_a_utf8_file_without_its_mark_is_read_as_utf8(tmp_path: Path) -> None:
    """Decoding it as cp1252 would not fail: it would turn "é" into "Ã©"."""
    row = shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "110", "2026/06/03", POLANCO)
    path = archive(tmp_path, one_fortnight("QQP_2026/06-2026_Q1.csv", row, encoding="utf-8"))

    prices = read_shelf_prices(path, "QQP_2026", PRODUCTS)

    assert prices["marca"].to_list() == ["Nescafé. Clásico"]


def test_a_coffee_product_nobody_listed_is_said_not_dropped_in_silence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    read_shelf_prices(archive(tmp_path, QQP_FORTNIGHTS), "QQP_2026", PRODUCTS)

    assert "the coffee category also lists ['Café en Cápsulas']" in caplog.text
    assert "Cafeteras" not in caplog.text  # another category altogether


def test_a_date_outside_its_fortnight_stops_the_read(tmp_path: Path) -> None:
    """A day and a month read the wrong way round land in another fortnight."""
    late = shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "110", "2026/06/20", POLANCO)
    path = archive(tmp_path, one_fortnight("QQP_2026/06-2026_Q1.csv", late))

    with pytest.raises(ValueError, match=r"1 rows are not dated inside 2026-06-01 - 2026-06-15"):
        read_shelf_prices(path, "QQP_2026", PRODUCTS)


def test_a_documented_column_that_is_missing_stops_the_read(tmp_path: Path) -> None:
    row = shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "110", "2026/06/03", POLANCO)
    columns = [c for c in QQP_COLUMNS if c != "latitud"]
    fortnights = {"QQP_2026/06-2026_Q1.csv": ("utf-8-sig", columns, [row[:-2] + row[-1:]])}

    with pytest.raises(ValueError, match=r"lacks the documented columns \['latitud'\]"):
        read_shelf_prices(archive(tmp_path, fortnights), "QQP_2026", PRODUCTS)


def test_an_archive_laid_out_otherwise_is_refused(tmp_path: Path) -> None:
    row = shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "110", "2026/06/03", POLANCO)

    with pytest.raises(ValueError, match="No CSV under 'QQP_2027'"):
        read_shelf_prices(archive(tmp_path, QQP_FORTNIGHTS), "QQP_2027", PRODUCTS)
    with pytest.raises(ValueError, match=re.escape("not named <month>-<year>_Q<1|2>.csv")):
        read_shelf_prices(
            archive(tmp_path, one_fortnight("QQP_2026/junio.csv", row)), "QQP_2026", PRODUCTS
        )


@pytest.mark.parametrize(
    ("member", "first", "last"),
    [
        ("QQP_2026/05-2026_Q1.csv", date(2026, 5, 1), date(2026, 5, 15)),
        ("QQP_2026/05-2026_Q2.csv", date(2026, 5, 16), date(2026, 5, 31)),
        ("QQP_2026/02-2026_Q2.csv", date(2026, 2, 16), date(2026, 2, 28)),
    ],
)
def test_a_file_names_its_fortnight(member: str, first: date, last: date) -> None:
    assert fortnight(member) == (first, last)


def test_a_date_is_read_either_way_the_files_write_it() -> None:
    dates = pl.DataFrame({"d": ["2026/05/04", "04/05/2026", "May 4"]}).select(
        record_date(pl.col("d"))
    )

    assert dates["d"].to_list() == [date(2026, 5, 4), date(2026, 5, 4), None]
