"""SQL access to every layer through DuckDB views over the latest Parquet partitions.

The connection is in-memory and read-only by construction: it holds views, never
data, so any number of readers (CLI, API, agent, UI) can open one while the
pipeline writes new partitions.

The tables another domain lends (`DomainConfig.uses`) are a catalog of their own, named
after it - `SELECT * FROM <domain>.clean.<table>` - holding only the tables declared.
"""

from collections.abc import Sequence
from pathlib import Path

import duckdb

from mlops_core.config import DATA_LAYERS, DomainUse
from mlops_core.storage import latest_partition, table_path

LAYERS = DATA_LAYERS


def connect(data_dir: Path, uses: Sequence[DomainUse] = ()) -> duckdb.DuckDBPyConnection:
    """One schema per layer, one view per table: `SELECT * FROM clean.<table>`; and each
    used domain's declared tables as `<domain>.<layer>.<table>`."""
    con = duckdb.connect()
    for layer in LAYERS:
        layer_dir = data_dir / layer
        if not layer_dir.is_dir():
            continue
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {layer}")
        for table_dir in sorted(p for p in layer_dir.iterdir() if p.is_dir()):
            partition = latest_partition(table_dir)
            parquet_path = partition / f"{table_dir.name}.parquet" if partition else None
            if parquet_path is None or not parquet_path.is_file():
                continue  # never built, or not a table: the analysis layer keeps its figures
            parquet = parquet_path.as_posix()
            # hive_partitioning off: the `built_at=` folder is lineage, not a data column.
            con.execute(
                f"CREATE VIEW {layer}.{table_dir.name} AS "
                f"SELECT * FROM read_parquet('{parquet}', hive_partitioning = false)"
            )
    for use in uses:
        con.execute(f"ATTACH ':memory:' AS {use.domain}")
        for name in use.qualified():
            domain, layer, table = name.split(".")
            partition = latest_partition(table_path(data_dir, name))
            if partition is None:
                continue  # the other domain has not built it yet: not offered
            parquet = (partition / f"{table}.parquet").as_posix()
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {domain}.{layer}")
            con.execute(
                f"CREATE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{parquet}', hive_partitioning = false)"
            )
    return con
