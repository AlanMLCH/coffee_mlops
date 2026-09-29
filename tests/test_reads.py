from datetime import UTC, datetime

import polars as pl

from domains.coffee.reads import READ_AT, daily_reads, newest

MORNING, EVENING = datetime(2026, 9, 1, 9, tzinfo=UTC), datetime(2026, 9, 1, 18, tzinfo=UTC)
LATER = datetime(2026, 9, 5, 9, tzinfo=UTC)
STACKED = pl.DataFrame({"place": ["a", "b", "a", "a"], READ_AT: [MORNING, MORNING, EVENING, LATER]})


def test_a_day_read_twice_is_its_later_read() -> None:
    reads = daily_reads(STACKED, LATER)

    assert [(at, frame.height) for at, frame in reads] == [(EVENING, 1), (LATER, 1)]
    assert READ_AT not in reads[0][1].columns


def test_the_newest_read_alone() -> None:
    assert newest(STACKED).to_dicts() == [{"place": "a"}]


def test_a_frame_without_the_column_is_a_single_read() -> None:
    single = pl.DataFrame({"place": ["a"]})

    assert daily_reads(single, LATER) == [(LATER, single)]
    assert newest(single) is single
