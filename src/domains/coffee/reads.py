"""The reads of a source that keeps every download (`accumulate` in the config).

The raw layer stacks each download's rows with the time it was read (`ingested_at`). A
clean table wants either the newest read (the catalogue, or the register, as it is now)
or one read per day (as it was at each). A frame without the column is a single read,
handed over as it is.

A download identical to the one before leaves no partition, so a read here is a day the
source *changed*; the days it was checked and found the same are in `checks.jsonl`.
"""

from datetime import datetime

import polars as pl

READ_AT = "ingested_at"  # the column each read's rows carry, from the raw layer


def daily_reads(frame: pl.DataFrame, read_at: datetime) -> list[tuple[datetime, pl.DataFrame]]:
    """Each day's latest read, oldest first, without the read column.

    One per day: a day read twice is its later read, which is the only way a correction
    can arrive. `read_at` dates a frame that is a single read.
    """
    if READ_AT not in frame.columns:
        return [(read_at, frame)]
    by_day: dict[object, datetime] = {}
    for at in sorted(frame[READ_AT].unique().to_list()):
        by_day[at.date()] = at  # a later read of the day replaces an earlier one
    return [(at, frame.filter(pl.col(READ_AT) == at).drop(READ_AT)) for at in by_day.values()]


def newest(frame: pl.DataFrame) -> pl.DataFrame:
    """The latest read alone, without the read column."""
    if READ_AT not in frame.columns:
        return frame
    return frame.filter(pl.col(READ_AT) == pl.col(READ_AT).max()).drop(READ_AT)
