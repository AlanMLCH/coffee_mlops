from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from coffee_mlops.storage import (
    latest_partition,
    prune_layers,
    prune_partitions,
    read_table,
    write_table,
)

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


def partition_of(table: Path, at: datetime) -> Path:
    return table / f"built_at={at.strftime('%Y%m%dT%H%M%SZ')}"


def test_pruning_keeps_the_newest_partitions(tmp_path: Path) -> None:
    table = tmp_path / "reviews"
    for day in range(1, 5):
        write_table(pl.DataFrame({"v": [day]}), table, inputs={}, at=T0.replace(day=day))

    deleted = prune_partitions(table, keep=2)

    assert len(deleted) == 2
    assert sorted(p.name for p in table.iterdir()) == [
        partition_of(table, T0.replace(day=3)).name,
        partition_of(table, T0.replace(day=4)).name,
    ]
    assert read_table(table)["v"].to_list() == [4]  # the newest still reads


def test_pruning_waits_before_deleting_a_partition_that_may_be_in_progress(
    tmp_path: Path,
) -> None:
    table = tmp_path / "reviews"
    write_table(pl.DataFrame({"v": [1]}), table, inputs={}, at=T0)
    in_progress = partition_of(table, T1)
    in_progress.mkdir()  # no manifest yet: a writer could be filling it right now

    assert prune_partitions(table, keep=1, stale_after=timedelta(days=1)) == []
    assert in_progress.is_dir()


def test_pruning_removes_a_build_that_crashed_long_ago(tmp_path: Path) -> None:
    table = tmp_path / "reviews"
    write_table(pl.DataFrame({"v": [1]}), table, inputs={}, at=T0)
    abandoned = partition_of(table, T1)
    abandoned.mkdir()

    deleted = prune_partitions(table, keep=5, now=T1 + timedelta(days=30))

    assert deleted == [abandoned]
    assert not abandoned.exists()


def test_pruning_walks_every_layer_and_table(tmp_path: Path) -> None:
    for layer, table in (("clean", "coffee_reviews"), ("features", "review_features")):
        for day in (1, 2):
            write_table(
                pl.DataFrame({"v": [day]}),
                tmp_path / layer / table,
                inputs={},
                at=T0.replace(day=day),
            )

    pruned = prune_layers(tmp_path, keep=1)

    assert pruned == {"clean/coffee_reviews": 1, "features/review_features": 1}
