"""INEGI's household income and expenditure survey (ENIGH): what households spend on coffee
to drink at home, and who they are.

Two of the survey's files, each a CSV inside its own ZIP, verified 2026-10-01 for 2024:

- `gastoshogar.csv` (54 MB zipped, 579 MB inside): every purchase the 91,414 households of
  the sample reported, 5,311,497 rows. A purchase names the survey's product code
  (`clave`), its type (`tipo_gasto`: G1 paid for the household, G3 from its own harvest),
  its value in the quarter (`gasto_tri` when paid, `gas_nm_tri` when not, at August 2024
  prices) and the household's state. Only the domain's coffee codes are kept: 17,356 rows.
- `concentradohogar.csv` (13 MB, 45 MB): a row per household with its weight (`factor`,
  the households it stands for: 38,830,230 in all), where it is (`ubica_geo`, state and
  municipality), the survey's stratum and primary sampling unit, how many live in it and
  its current income in the quarter (`ing_cor`; the weighted mean is 77,864 pesos).

Both plain ASCII, keys with their leading zeros: read as text, typed by the contract.
"""

import zipfile
from collections.abc import Collection
from pathlib import Path

import polars as pl

SPENDING_COLUMNS = ["folioviv", "foliohog", "clave", "tipo_gasto", "gasto_tri", "gas_nm_tri",
                    "entidad"]  # fmt: skip
HOUSEHOLD_COLUMNS = ["folioviv", "foliohog", "ubica_geo", "est_dis", "upm", "factor",
                     "tot_integ", "ing_cor"]  # fmt: skip


def read_spending(path: Path, member: str, codes: Collection[str]) -> pl.DataFrame:
    """The survey's purchases of the products named by `codes`, as text."""
    return _read(path, member, SPENDING_COLUMNS).filter(pl.col("clave").is_in(list(codes)))


def read_households(path: Path, member: str) -> pl.DataFrame:
    """Each household of the survey: its weight, place, sample design, size and income."""
    return _read(path, member, HOUSEHOLD_COLUMNS)


def _read(path: Path, member: str, columns: list[str]) -> pl.DataFrame:
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    header = pl.read_csv(data, infer_schema_length=0, n_rows=0).columns
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"{member} has no {missing}: INEGI changed its layout")
    # A value left blank is written as a space ("gasto_tri" of a gift): null, not text.
    frame = pl.read_csv(data, columns=columns, infer_schema_length=0)
    return frame.with_columns(pl.all().str.strip_chars().replace("", None))
