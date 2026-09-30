"""PROFECO's *Quién es Quién en los Precios*: what packaged coffee costs on the shelf.

PROFECO's staff price a basket of products in supermarkets, convenience stores, markets
and pharmacies across Mexico, and publish each year as one archive of fortnightly CSVs:
the year in course a ZIP (`QQP_2026/05-2026_Q1.csv` is the first half of May), a closed
year a RAR 5 (`QQP_2025/05-2025_01.csv`), which the core opens with bsdtar
(`mlops_core.data.archives`). Seven months of 2026 are 2.5 GB unpacked, and 64,451 of
those rows are coffee; 2024 and 2025 add 117,981 and 98,948 (verified 2026-09-29, every
fortnight of both: UTF-8 with a mark, the fifteen documented columns, every date inside
its fortnight). This reader keeps the products the YAML names and the fifteen columns
PROFECO's data dictionary documents, and nothing else.

What the files do not say, and the reader has to know (verified 2026-09-27):

- **The character set changes from file to file.** UTF-8 with a byte-order mark, except
  May's two, in cp1252 without one. A file that does not decode as UTF-8 is read as
  cp1252, and the log says which.
- **So do the columns.** June's files add three the dictionary does not document
  (`folio`, `cv_producto`, `cv_marca`). They are left out; a documented one missing
  stops the read.
- **And the date format.** May writes `dd/mm/yyyy`, the rest `yyyy/mm/dd`. Every date
  is checked to fall inside the fortnight its file is named for, so a day and a month
  read the wrong way round cannot pass: all 64,451 do.
- **Categories are spelled two ways** ("Cafe"/"Café", "Basicos"/"Básicos"), so coffee is
  found by its product name, which is spelled one way. A product that shows up in the
  coffee category and is not in the YAML is said in the log, not silently dropped.
"""

import calendar
import logging
import re
from collections.abc import Collection
from datetime import date
from pathlib import Path

import polars as pl

from mlops_core.data import archives

logger = logging.getLogger(__name__)

# PROFECO's data dictionary (`QQP_diccionario_dataset.csv`), in its order.
COLUMNS = ["producto", "presentacion", "marca", "categoria", "catalogo", "precio",
           "fecha_registro", "cadena_comercial", "giro", "nombre_comercial", "direccion",
           "estado", "municipio", "latitud", "longitud"]  # fmt: skip
FILE = "file"  # the member each row was read from
# `05-2026_Q2.csv` in the year in course, `05-2025_02.csv` in a closed one.
_FORTNIGHT = re.compile(r"(?P<month>\d{2})-(?P<year>\d{4})_(?:Q|0)(?P<half>[12])\.csv$")
_BOM = b"\xef\xbb\xbf"


def read_shelf_prices(path: Path, folder: str, products: Collection[str]) -> pl.DataFrame:
    """Every fortnight in `folder` of the archive, as text: the rows of `products`, the
    documented columns, and the file each row came from."""
    members = sorted(
        name
        for name in archives.names(path)
        if name.startswith(folder.rstrip("/") + "/") and name.endswith(".csv")
    )
    if not members:
        raise ValueError(f"No CSV under {folder!r} in {path.name}: the archive changed")
    return pl.concat([_read_fortnight(path, member, products) for member in members])


def record_date(text: pl.Expr) -> pl.Expr:
    """A `fecha_registro` as a date, in either of the two ways the files write it."""
    return pl.coalesce(
        text.str.to_date("%Y/%m/%d", strict=False), text.str.to_date("%d/%m/%Y", strict=False)
    )


def fortnight(member: str) -> tuple[date, date]:
    """First and last day of the fortnight a file is named for: `05-2026_Q2.csv` (or
    `05-2025_02.csv`) is 16-31 May."""
    found = _FORTNIGHT.search(member)
    if found is None:
        raise ValueError(
            f"{member}: not named <month>-<year>_Q<1|2>.csv or _0<1|2>.csv, as every fortnight is"
        )
    year, month = int(found["year"]), int(found["month"])
    if found["half"] == "1":
        return date(year, month, 1), date(year, month, 15)
    return date(year, month, 16), date(year, month, calendar.monthrange(year, month)[1])


def _read_fortnight(path: Path, member: str, products: Collection[str]) -> pl.DataFrame:
    first, last = fortnight(member)  # before reading 200 MB: a stray file fails fast
    data = _utf8(archives.read(path, member), member)
    header = pl.read_csv(data, n_rows=0, infer_schema_length=0).columns
    missing = [column for column in COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{member} lacks the documented columns {missing}")
    undocumented = [column for column in header if column not in COLUMNS]
    if undocumented:
        logger.info("%s: leaving out undocumented columns %s", member, undocumented)
    rows = pl.read_csv(data, columns=COLUMNS, infer_schema_length=0)
    _say_what_is_left_out(rows, products, member)
    kept = rows.filter(pl.col("producto").is_in(list(products)))
    dates = kept.select(record_date(pl.col("fecha_registro")).alias("date"))["date"]
    outside = int((dates.is_null() | (dates < first) | (dates > last)).sum())
    if outside:
        raise ValueError(
            f"{member}: {outside} rows are not dated inside {first} - {last}: "
            "the date format changed, or the file is not the fortnight it says"
        )
    return kept.with_columns(pl.lit(member).alias(FILE))


def _utf8(data: bytes, member: str) -> bytes:
    """The file as UTF-8. The mark is left on: polars reads past it."""
    if data.startswith(_BOM):
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        logger.info("%s is not UTF-8: read as cp1252", member)
        return data.decode("cp1252").encode("utf-8")
    return data


def _say_what_is_left_out(rows: pl.DataFrame, products: Collection[str], member: str) -> None:
    """Warn about a product filed under the same category as the ones read, that is not
    read: PROFECO adding, say, capsules would otherwise vanish without a word."""
    folded = (
        pl.col("categoria").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
    )
    listed = pl.col("producto").is_in(list(products))
    categories = rows.filter(listed).select(folded.unique())["categoria"]
    unread = rows.filter(folded.is_in(categories.to_list()) & ~listed)["producto"].unique()
    if not unread.is_empty():
        logger.warning(
            "%s: the coffee category also lists %s, which is not read: "
            "add it to `consumer_prices.products` if it is coffee",
            member,
            sorted(unread.to_list()),
        )
