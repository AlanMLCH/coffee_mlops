"""FAOSTAT's producer prices: what a country's farmers are paid for a crop, year by year.

FAO publishes every crop of every country in one bulk file,
`Prices_E_All_Data_(Normalized).csv` inside a ZIP: 1.3 million rows, 214 MB unpacked,
every field quoted, UTF-8 without a mark (verified 2026-09-29). Coffee is 6,291 of those
rows. Reading the whole file into a frame to keep 0.5% of it would hold ~2 GB of text in
memory, so this reader streams the member line by line, keeps the lines that could be the
item (its code appears quoted in the line) and lets the frame keep the ones that are: a
value that happens to be "656" passes the first test and fails the second. About a second.
"""

import io
import zipfile
from pathlib import Path

import polars as pl

# The columns the contract reads, in FAO's names. The others (the M49 and CPC codes, the
# year and month codes, the element's name) repeat what these say.
COLUMNS = ["Area Code", "Area", "Item Code", "Item", "Element Code", "Element", "Year",
           "Months", "Unit", "Value", "Flag"]  # fmt: skip


def read_producer_prices(path: Path, member: str, item_code: str) -> pl.DataFrame:
    """The rows of one item from FAOSTAT's bulk file of producer prices, as text."""
    quoted = f',"{item_code}",'
    with zipfile.ZipFile(path) as archive, archive.open(member) as raw:
        lines = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        header = next(lines, "")
        kept = [line for line in lines if quoted in line]
    if not header.startswith("Area Code"):
        raise ValueError(f"{member} in {path.name} does not start with FAOSTAT's header")
    frame = pl.read_csv(io.StringIO(header + "".join(kept)), infer_schema_length=0)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{member} has no {missing}: FAOSTAT changed its layout")
    return frame.filter(pl.col("Item Code") == item_code).select(COLUMNS)
