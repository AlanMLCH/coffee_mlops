"""SQL access to every layer through DuckDB views over the latest Parquet partitions.

The connection is in-memory and read-only by construction: it holds views, never
data, so any number of readers (CLI, API, agent, UI) can open one while the
pipeline writes new partitions.
"""

from pathlib import Path

import duckdb

from coffee_mlops.storage import latest_partition

LAYERS = ("clean", "features", "predictions", "analysis")


def connect(data_dir: Path) -> duckdb.DuckDBPyConnection:
    """One schema per layer, one view per table: `SELECT * FROM clean.coffee_reviews`."""
    con = duckdb.connect()
    for layer in LAYERS:
        layer_dir = data_dir / layer
        if not layer_dir.is_dir():
            continue
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {layer}")
        for table_dir in sorted(p for p in layer_dir.iterdir() if p.is_dir()):
            partition = latest_partition(table_dir)
            if partition is None:
                continue
            parquet = (partition / f"{table_dir.name}.parquet").as_posix()
            # hive_partitioning off: the `built_at=` folder is lineage, not a data column.
            con.execute(
                f"CREATE VIEW {layer}.{table_dir.name} AS "
                f"SELECT * FROM read_parquet('{parquet}', hive_partitioning = false)"
            )
    return con
