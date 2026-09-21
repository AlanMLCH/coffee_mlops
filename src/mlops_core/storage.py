"""Immutable, timestamped partitions shared by every layer (raw, clean, features, ...).

    <table_dir>/<key>=<UTC timestamp>/<files> + manifest.json

A partition is written once and never modified. The manifest is written last, so
a partition without one is incomplete and ignored. Readers always take the newest
complete partition: a writer never blocks or corrupts a reader.
"""

import logging
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
# Microseconds, not seconds: two builds inside the same second are rare but real (a
# test, a retry, a fast loop), and colliding on a partition name crashed the run.
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"


class TableManifest(BaseModel):
    table: str
    rows: int
    built_at: datetime
    # Upstream name -> the partition this table was built from (lineage).
    inputs: dict[str, str]


def new_partition(table_dir: Path, key: str, at: datetime) -> Path:
    partition = table_dir / f"{key}={at.strftime(TIMESTAMP_FORMAT)}"
    partition.mkdir(parents=True)
    return partition


def latest_partition(table_dir: Path) -> Path | None:
    complete = sorted(p for p in table_dir.glob("*=*") if (p / MANIFEST_NAME).is_file())
    return complete[-1] if complete else None


def write_table(
    df: pl.DataFrame,
    table_dir: Path,
    inputs: Mapping[str, str],
    at: datetime | None = None,
    also_csv: bool = False,
) -> Path:
    """Write one partition. `also_csv` adds a CSV copy for tables a human opens in a
    spreadsheet; Parquet stays the one readers and the catalog use."""
    built_at = at or datetime.now(UTC)
    partition = new_partition(table_dir, "built_at", built_at)
    path = partition / f"{table_dir.name}.parquet"
    df.write_parquet(path)
    if also_csv:
        df.write_csv(partition / f"{table_dir.name}.csv")  # before the manifest, like the Parquet
    manifest = TableManifest(
        table=table_dir.name, rows=df.height, built_at=built_at, inputs=dict(inputs)
    )
    (partition / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))
    return path


def read_table(table_dir: Path) -> pl.DataFrame:
    partition = latest_partition(table_dir)
    if partition is None:
        raise FileNotFoundError(f"No complete partition of '{table_dir.name}' in {table_dir}")
    return pl.read_parquet(partition / f"{table_dir.name}.parquet")


def prune_partitions(
    table_dir: Path,
    keep: int,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(days=1),
) -> list[Path]:
    """Delete old partitions of one table, keeping the newest `keep` complete ones.

    History is worth keeping (it is how a past prediction stays explainable) but not
    forever. Incomplete partitions are only removed once they are older than
    `stale_after`: a younger one may belong to a writer that is still running.
    """
    partitions = sorted(table_dir.glob("*=*"))
    complete = [p for p in partitions if (p / MANIFEST_NAME).is_file()]
    abandoned = [
        p for p in partitions if p not in complete and _age(p, now) >= stale_after
    ]  # a writer that crashed long enough ago that nobody is filling it
    superseded = complete[: max(len(complete) - keep, 0)]  # sorted oldest first
    deleted = abandoned + superseded
    for partition in deleted:
        shutil.rmtree(partition)
        logger.info("pruned %s", partition)
    return deleted


def prune_layers(data_dir: Path, keep: int, now: datetime | None = None) -> dict[str, int]:
    """Prune every table of every layer under a domain's data dir."""
    pruned = {}
    for layer in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for table_dir in sorted(p for p in layer.iterdir() if p.is_dir()):
            removed = prune_partitions(table_dir, keep, now)
            if removed:
                pruned[f"{layer.name}/{table_dir.name}"] = len(removed)
    return pruned


def _age(partition: Path, now: datetime | None) -> timedelta:
    modified = datetime.fromtimestamp(partition.stat().st_mtime, tz=UTC)
    return (now or datetime.now(UTC)) - modified
