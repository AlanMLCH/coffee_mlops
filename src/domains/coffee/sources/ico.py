"""The ICO's daily indicator prices, read from the one-page PDF it publishes.

`https://ico.org/documents/I-CIP.pdf` is the current month and nothing else: one row per
trading day (the composite I-CIP and the four group indicators, in US cents per pound),
blank rows for the days still to come, then the month's Average, High and Low. So the
history is every download stacked (the source is `accumulate`), and each download has to
be read right: a PDF table is the kind of figure this project does not trust
(CLAUDE.md, corpus: "las cifras las responde el SQL, nunca un PDF").

This one earns the exception because it checks itself. The page states its own
Average, High and Low; a reading whose days do not average, top and bottom out to them
is refused rather than stored, and so is a page whose columns are not the five this
reader knows, in that order.
"""

import re
from pathlib import Path

import polars as pl

# The columns, in the page's order, and how the page heads them (its header wraps over
# three lines; the words, in order, are what is checked).
INDICATORS = ("i_cip", "colombian_milds", "other_milds", "brazilian_naturals", "robustas")
HEADER = "I-CIP Colombian Milds Other Milds Brazilian Naturals Robustas"
COLUMNS = ("date", *INDICATORS)

_TITLE = re.compile(r"ICO Indicator Prices - (?P<month>[A-Z][a-z]+) (?P<year>\d{4}) \(I-CIP\)")
_UNIT = "In US cents/lb"
_NUMBER = r"(\d+(?:\.\d+)?)"
_DAY = re.compile(rf"^(\d{{1,2}})-([A-Z][a-z]{{2}})\s+{r'\s+'.join([_NUMBER] * 5)}\s*$")
_BLANK_DAY = re.compile(r"^\d{1,2}-[A-Z][a-z]{2}\s*$")
_SUMMARY = re.compile(rf"^(Average|High|Low)\s+{r'\s+'.join([_NUMBER] * 5)}\s*$")
# The page rounds its average to the cent; a mean within half a cent of it agrees.
ROUNDING = 0.005 + 1e-9
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")  # fmt: skip


def read_indicator_prices(path: Path) -> pl.DataFrame:
    """The PDF's days, as text columns: `date` (ISO) and the five indicators."""
    from pypdf import PdfReader  # the data extra's; imported where used

    reader = PdfReader(path)
    return parse_indicator_prices("\n".join(page.extract_text() or "" for page in reader.pages))


def parse_indicator_prices(text: str) -> pl.DataFrame:
    """The days of the page's text, checked against its own summary rows."""
    lines = [line.strip() for line in text.splitlines()]
    title = next((m for m in map(_TITLE.search, lines) if m), None)
    if title is None or _UNIT not in lines:
        raise ValueError("Not the ICO's indicator price page: no title, or not in US cents/lb")
    month, year = MONTHS.index(title["month"]) + 1, int(title["year"])
    start = lines.index(_UNIT) + 1
    first_day = next(
        (i for i, line in enumerate(lines) if _DAY.match(line) or _BLANK_DAY.match(line)), None
    )
    if first_day is None:
        raise ValueError("No row of days on the page, not even blank ones: the layout changed")
    header = " ".join(" ".join(lines[start:first_day]).split())
    if header != HEADER:
        raise ValueError(f"The columns changed: {header!r}, not {HEADER!r}")

    rows, summary = [], {}
    for line in lines[first_day:]:
        if day := _DAY.match(line):
            if day[2] != title["month"][:3]:
                raise ValueError(f"A {day[2]} row on the {title['month']} page: {line!r}")
            rows.append((f"{year:04d}-{month:02d}-{int(day[1]):02d}", *day.groups()[2:]))
        elif found := _SUMMARY.match(line):
            summary[found[1]] = [float(value) for value in found.groups()[1:]]
    frame = pl.DataFrame(rows, schema=dict.fromkeys(COLUMNS, pl.String), orient="row")
    if rows:
        _agrees_with_its_summary(frame, summary)
    return frame


def _agrees_with_its_summary(frame: pl.DataFrame, summary: dict[str, list[float]]) -> None:
    missing = sorted({"Average", "High", "Low"} - set(summary))
    if missing:
        raise ValueError(f"The page has days but no {missing} row to check them against")
    for n, indicator in enumerate(INDICATORS):
        values = frame[indicator].cast(pl.Float64)
        read = {"Average": values.mean(), "High": values.max(), "Low": values.min()}
        for row, value in read.items():
            got = float(value)  # type: ignore[arg-type]
            if abs(got - summary[row][n]) > ROUNDING:
                raise ValueError(
                    f"{indicator}: the days read give {row} {got:.3f}, the page says "
                    f"{summary[row][n]:.2f}: the page was misread"
                )
