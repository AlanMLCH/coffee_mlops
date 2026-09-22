"""Features layer: the domain's items, enriched with their context -> model-ready table.

Only stateless, row-wise transforms happen here. Anything fitted on data (encoders,
rare-category grouping, imputation) lives in the model pipeline, so it is learned on
the training split only and ships with the model to serving. What an item is allowed to
know about its context is the domain's `enrich`, the same function the API calls.
"""

from datetime import UTC, datetime
from pathlib import Path

import pandera.polars as pa
import polars as pl

from mlops_core.adapter import DomainAdapter
from mlops_core.config import ItemsConfig, ModelSpec
from mlops_core.contracts import check_contract
from mlops_core.storage import latest_partition, read_table, write_table


def features_schema(items: ItemsConfig, spec: ModelSpec) -> pa.DataFrameSchema:
    """The feature table's contract, derived from the domain's config: keys, the declared
    features with their types, and the target - nothing else, so a leaking column
    cannot ride along."""
    return pa.DataFrameSchema(
        name=items.features_table,
        strict=True,
        unique=[items.id],
        columns={
            items.id: pa.Column(pl.String),
            items.period: pa.Column(pl.String),
            items.time: pa.Column(pl.Date),
            **{c: pa.Column(pl.String, nullable=True) for c in spec.categorical},
            **{c: pa.Column(pl.Float64, nullable=True) for c in spec.numeric},
            spec.target: pa.Column(pl.Float64),
        },
    )


def select_features(enriched: pl.DataFrame, items: ItemsConfig, spec: ModelSpec) -> pl.DataFrame:
    """Keys, features and target, in that order. Every other column is dropped here -
    leakage included - so no downstream consumer can pick one up."""
    return enriched.select(
        *items.keys,
        *spec.categorical,
        *[pl.col(c).cast(pl.Float64) for c in spec.numeric],
        spec.target,
    )


def build_features(adapter: DomainAdapter, data_dir: Path, at: datetime | None = None) -> Path:
    """Read the latest items and context, enrich, check the contract, write Parquet."""
    config = adapter.config
    items = config.items
    clean_dir = data_dir / "clean"
    inputs = (items.table, *adapter.context_tables())
    lineage = {}
    for table in inputs:
        partition = latest_partition(clean_dir / table)
        lineage[table] = partition.name if partition else ""
    context = {table: read_table(clean_dir / table) for table in adapter.context_tables()}
    enriched = adapter.enrich(read_table(clean_dir / items.table), context)
    features = check_contract(
        features_schema(items, config.model), select_features(enriched, items, config.model)
    )
    return write_table(
        features, data_dir / "features" / items.features_table, lineage, at or datetime.now(UTC)
    )
