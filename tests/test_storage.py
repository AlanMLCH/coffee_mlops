from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from mlops_core.storage import (
    TIMESTAMP_FORMAT,
    built_from,
    data_version,
    latest_data_version,
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
    partition_of(table, T1).mkdir()  # a writer crashed mid-way

    assert latest_partition(table) == partition_of(table, T0)


def test_reading_a_table_never_built_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reviews"):
        read_table(tmp_path / "reviews")


def partition_of(table: Path, at: datetime) -> Path:
    """The name `write_table` gives a partition, derived from the module's own format
    so these tests never pin a timestamp layout the code is free to change."""
    return table / f"built_at={at.strftime(TIMESTAMP_FORMAT)}"


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


# --- The data version -------------------------------------------------------------------


def clean_build(data_dir: Path, raw: dict[str, str], at: datetime) -> str:
    """A clean table built from `raw` partitions; its partition's name."""
    path = write_table(pl.DataFrame({"v": [1]}), data_dir / "clean" / "lots", raw, at=at)
    return path.parent.name


def test_the_data_version_follows_the_raw_data_not_the_rebuilds(tmp_path: Path) -> None:
    """A clean build writes a new partition whatever happened; the version changes only
    when the raw partitions behind it do."""
    first = clean_build(tmp_path, {"cqi": "ingested_at=A"}, T0)
    rebuilt = clean_build(tmp_path, {"cqi": "ingested_at=A"}, T1)
    changed = clean_build(tmp_path, {"cqi": "ingested_at=B"}, T1 + timedelta(hours=1))

    same = data_version(tmp_path, {"lots": first})
    assert data_version(tmp_path, {"lots": rebuilt}) == same
    assert data_version(tmp_path, {"lots": changed}) != same
    assert latest_data_version(tmp_path, ["lots"]) == data_version(tmp_path, {"lots": changed})
    assert latest_data_version(tmp_path, ["lots", "never_built"]) is None
    # A clean partition since pruned stands for itself.
    assert data_version(tmp_path, {"lots": "built_at=pruned"}) != same


def test_a_derived_table_knows_the_data_it_was_built_from(tmp_path: Path) -> None:
    clean = clean_build(tmp_path, {"cqi": "ingested_at=A"}, T0)
    features = tmp_path / "features" / "lot_features"
    write_table(pl.DataFrame({"v": [1]}), features, {"lots": clean}, at=T0)

    assert built_from(tmp_path, features) == data_version(tmp_path, {"lots": clean})
    assert built_from(tmp_path, tmp_path / "features" / "never_built") is None
