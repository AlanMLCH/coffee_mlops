"""Immutable, timestamped partitions shared by every layer (raw, clean, features, ...).

    <table_dir>/<key>=<UTC timestamp>/<files> + manifest.json

A partition is written once and never modified. The manifest is written last, so
a partition without one is incomplete and ignored. Readers always take the newest
complete partition: a writer never blocks or corrupts a reader.
"""

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
RAW = "raw"  # the layer of downloads as they came: the record, never pruned
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


def data_version(data_dir: Path, clean: Mapping[str, str]) -> str:
    """Twelve characters that change when, and only when, the raw data behind some clean
    tables does.

    `clean` maps each table to the partition read. A clean build writes new partitions
    even when nothing changed, so their names say nothing; what each was built from does.
    Raw partitions are content-addressed - a download identical to the last one stores
    nothing - so the raw partitions behind a table name its data. A clean partition whose
    manifest is gone (pruned) stands for itself.
    """
    raw: dict[str, str] = {}
    for table, partition in sorted(clean.items()):
        manifest = table_path(data_dir, table) / partition / MANIFEST_NAME
        if not manifest.is_file():
            raw[table] = partition
            continue
        # Keyed by table as well: two tables may name the same source from different
        # downloads, and one must not hide the other.
        for source, lineage in TableManifest.model_validate_json(
            manifest.read_text()
        ).inputs.items():
            raw[f"{table}/{source}"] = lineage
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return digest[:12]


def table_path(data_dir: Path, name: str) -> Path:
    """Where a table lives: a bare name is one of the domain's clean tables; a qualified
    one (`<domain>.<layer>.<table>`) is another domain's, beside this one under the same
    data root. Whether the domain may read it is its config's to say (`DomainConfig.uses`)."""
    parts = name.split(".")
    if len(parts) == 3:
        domain, layer, table = parts
        return data_dir.parent / domain / layer / table
    return data_dir / "clean" / name


def latest_data_version(data_dir: Path, tables: Sequence[str]) -> str | None:
    """The data version of the newest partitions of `tables`; None until all exist."""
    partitions = {table: latest_partition(table_path(data_dir, table)) for table in tables}
    if any(partition is None for partition in partitions.values()):
        return None
    return data_version(data_dir, {t: p.name for t, p in partitions.items() if p is not None})


def rows_version(frame: pl.DataFrame) -> str:
    """Twelve characters that change when, and only when, a table's rows do - whatever
    order they were written in.

    What a model learns from is its feature table's rows, and that is what "trained on
    this data" has to mean. The raw data behind a table is too coarse for it: a table can
    stack several sources, and a new download of one the model does not read (a day of
    prices for a model of months) changed the raw version, and retrained a model on rows
    it had already learned from (2026-09-29). Row hashes are polars': stable within one
    version of it, so an upgrade may retrain each model once.
    """
    columns = sorted(frame.columns)
    hashes = frame.select(columns).hash_rows(seed=0).sort().to_list()
    payload = json.dumps([columns, [str(frame.schema[c]) for c in columns], hashes])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def content_version(table_dir: Path) -> str | None:
    """The rows version of a table's newest partition; None if it has none."""
    if latest_partition(table_dir) is None:
        return None
    return rows_version(read_table(table_dir))


def built_from(data_dir: Path, table_dir: Path) -> str | None:
    """The data version a derived table's newest partition was built from (the clean
    partitions its manifest lists); None if it has none."""
    partition = latest_partition(table_dir)
    if partition is None:
        return None
    manifest = TableManifest.model_validate_json((partition / MANIFEST_NAME).read_text())
    return data_version(data_dir, manifest.inputs)


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
    """Prune every table of every derived layer under a domain's data dir; never `raw/`.

    The raw layer is the record: every download that brought something new, as it came.
    A derived layer can be rebuilt from it, so its old builds can go; a download cannot
    always be made again - a page that shows only the current month, a catalogue that
    shows only today, a file its publisher has since replaced."""
    pruned = {}
    for layer in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != RAW):
        for table_dir in sorted(p for p in layer.iterdir() if p.is_dir()):
            removed = prune_partitions(table_dir, keep, now)
            if removed:
                pruned[f"{layer.name}/{table_dir.name}"] = len(removed)
    return pruned


def _age(partition: Path, now: datetime | None) -> timedelta:
    modified = datetime.fromtimestamp(partition.stat().st_mtime, tz=UTC)
    return (now or datetime.now(UTC)) - modified
