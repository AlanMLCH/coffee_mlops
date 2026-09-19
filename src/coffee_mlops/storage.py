"""Immutable, timestamped partitions shared by every layer (raw, clean, features, ...).

    <table_dir>/<key>=<UTC timestamp>/<files> + manifest.json

A partition is written once and never modified. The manifest is written last, so
a partition without one is incomplete and ignored. Readers always take the newest
complete partition: a writer never blocks or corrupts a reader.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic import BaseModel

MANIFEST_NAME = "manifest.json"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


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
    df: pl.DataFrame, table_dir: Path, inputs: Mapping[str, str], at: datetime | None = None
) -> Path:
    built_at = at or datetime.now(UTC)
    partition = new_partition(table_dir, "built_at", built_at)
    path = partition / f"{table_dir.name}.parquet"
    df.write_parquet(path)
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
