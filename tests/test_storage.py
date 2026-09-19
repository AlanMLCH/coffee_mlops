from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from coffee_mlops.storage import latest_partition, read_table, write_table

T0 = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def test_readers_get_the_newest_complete_partition(tmp_path: Path) -> None:
    table = tmp_path / "reviews"
    write_table(pl.DataFrame({"v": [1]}), table, inputs={}, at=T0)
    write_table(pl.DataFrame({"v": [2]}), table, inputs={}, at=T1)

    assert read_table(table)["v"].to_list() == [2]


def test_partitions_are_never_overwritten(tmp_path: Path) -> None:
    table = tmp_path / "reviews"
    write_table(pl.DataFrame({"v": [1]}), table, inputs={}, at=T0)

    with pytest.raises(FileExistsError):
        write_table(pl.DataFrame({"v": [2]}), table, inputs={}, at=T0)


def test_partition_without_manifest_is_invisible(tmp_path: Path) -> None:
    table = tmp_path / "reviews"
    write_table(pl.DataFrame({"v": [1]}), table, inputs={}, at=T0)
    (table / "built_at=20260920T120000Z").mkdir()  # a writer crashed mid-way

    assert latest_partition(table) == table / "built_at=20260919T120000Z"


def test_reading_a_table_never_built_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reviews"):
        read_table(tmp_path / "reviews")
