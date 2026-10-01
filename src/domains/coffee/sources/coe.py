"""Cup of Excellence Mexico: each year's winning lots and what their auction paid, read from
the Alliance for Coffee Excellence's page for the year.

`https://allianceforcoffeeexcellence.org/mexico-<year>/`, verified 2026-10-01 with the
project's client: robots.txt allows every agent; the site's own list of Mexico's
competitions has 2012-2015, 2017-2019 and 2021-2026 (no page for 2016 or 2020). Each page
is WordPress HTML with its tables written by hand, and no two eras lay them out alike:

- the **results**: rank ("1", "1a", "NW" for a national winner), score, farm, farmer,
  region; weight from 2019, variety and process from 2018 ("Proceso", "Variedad" in 2023;
  "Process, Variety" in one cell in 2019); 2024 ran three competitions, a table each, every
  one with its own "1A";
- the **auction**: lot or rank, farm (with the farmer, in 2018), score (from 2021), weight
  in pounds or boxes, the high bid in US dollars a pound ("$50.21/lb", "US$ 61.00",
  "75.10"), the total and the buyers; the national winners have their own pair of tables;
- in 2017-2019 the last lots sit in one table holding both.

The reader keeps every table that names a farm and a score or a bid, a row each, its
columns named by what their header says and every value as text: a row of the results is
a lot, a row with a bid is a sale, a row of both is both. Judges, sponsors and commissions
are left out. Pairing lots with their sales, and typing them, is the clean layer's.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

import polars as pl

# A header -> the column it is, by the first pattern it matches (lower-cased, spaces
# collapsed). Order matters: "farmer" before "farm", "high bidder(s)" before "high bid".
HEADERS = [
    ("farmer", r"farmer"),
    ("farm", r"farm"),
    ("buyers", r"bidder|company name|winners"),
    ("bid", r"bid"),
    ("total", r"total value"),
    ("process_variety", r"process, variety"),
    ("process", r"process|proceso"),
    ("variety", r"variet|variedad"),
    ("rank", r"^rank$|^lot #$"),
    ("score", r"score"),
    ("region", r"region"),
    ("weight_kg", r"\(kg\)"),
    ("weight_lb", r"\(lbs?\.?\)|lot lbs"),
]
COLUMNS = ["year", "table", "rank", "score", "farm", "farmer", "region", "variety", "process",
           "weight_kg", "weight_lb", "bid", "total", "buyers"]  # fmt: skip


def read_competition(path: Path) -> pl.DataFrame:
    """Every row of the page's tables of lots and sales, as text; the year from the file's
    name (`mexico-<year>.html`)."""
    found = re.search(r"(\d{4})", path.name)
    if found is None:
        raise ValueError(f"{path.name} names no year")
    parser = _Tables()
    parser.feed(path.read_text(encoding="utf-8"))
    rows = []
    for number, table in enumerate(parser.tables):
        if not table:
            continue
        header = [_column(cell) for cell in table[0]]
        if "farm" not in header or not {"score", "bid"} & set(header):
            continue  # judges, sponsors, commissions
        for cells in table[1:]:
            if len(cells) != len(header) or not any(cells):
                continue  # a row spanning the table: a heading inside it
            row = {name: value or None for name, value in zip(header, cells, strict=True)
                   if name is not None}  # fmt: skip
            if not row.get("farm"):
                continue  # "Totals:", "Stats:" - the sums under an auction
            if "process_variety" in row:  # "NATURAL, Borbón"
                process, _, variety = (row.pop("process_variety") or "").partition(",")
                row |= {"process": process.strip() or None, "variety": variety.strip() or None}
            rows.append(row | {"year": found.group(1), "table": str(number)})
    if not rows:
        raise ValueError(f"{path.name} has no table of lots: the page changed its layout")
    return pl.DataFrame(rows, schema=dict.fromkeys(COLUMNS, pl.String)).select(COLUMNS)


def _column(header: str) -> str | None:
    text = " ".join(header.lower().split())
    return next((name for name, pattern in HEADERS if re.search(pattern, text)), None)


class _Tables(HTMLParser):
    """Each table's rows of cell texts, whitespace collapsed."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
