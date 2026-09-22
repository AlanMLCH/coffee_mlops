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
from mlops_core.config import GroupSplit, ModelConfig
from mlops_core.contracts import check_contract
from mlops_core.storage import latest_partition, read_table, write_table


def features_schema(model: ModelConfig) -> pa.DataFrameSchema:
    """The feature table's contract, derived from the model's config: keys, the declared
    features with their types, and the target - nothing else, so a leaking column
    cannot ride along."""
    items, spec, split = model.items, model.spec, model.training.split
    group = {split.column: pa.Column(pl.String)} if isinstance(split, GroupSplit) else {}
    return pa.DataFrameSchema(
        name=model.features_table,
        strict=True,
        unique=[items.id],
        columns={
            items.id: pa.Column(pl.String),
            items.period: pa.Column(pl.String),
            items.time: pa.Column(pl.Date),
            **group,
            **{c: pa.Column(pl.String, nullable=True) for c in spec.categorical},
            **{c: pa.Column(pl.Float64, nullable=True) for c in spec.numeric},
            spec.target: pa.Column(pl.Float64),
        },
    )


def select_features(enriched: pl.DataFrame, model: ModelConfig) -> pl.DataFrame:
    """Keys, features and target, in that order. Every other column is dropped here -
    leakage included - so no downstream consumer can pick one up."""
    spec = model.spec
    return enriched.select(
        *model.keys,
        *spec.categorical,
        *[pl.col(c).cast(pl.Float64) for c in spec.numeric],
        spec.target,
    )


def build_features(
    adapter: DomainAdapter, model_name: str, data_dir: Path, at: datetime | None = None
) -> Path:
    """Read the model's latest items and context, enrich, check the contract, write Parquet."""
    model = adapter.config.model_named(model_name)
    clean_dir = data_dir / "clean"
    context_tables = adapter.context_tables(model.name)
    lineage = {}
    for table in (model.items.table, *context_tables):
        partition = latest_partition(clean_dir / table)
        lineage[table] = partition.name if partition else ""
    context = {table: read_table(clean_dir / table) for table in context_tables}
    enriched = adapter.enrich(model.name, read_table(clean_dir / model.items.table), context)
    features = check_contract(features_schema(model), select_features(enriched, model))
    return write_table(
        features, data_dir / "features" / model.features_table, lineage, at or datetime.now(UTC)
    )
