"""Batch inference.

Scores the whole feature table with the champion and writes the result as another
immutable Parquet partition, so predictions are queryable next to the data that
produced them and a running API or agent is never blocked by the job.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandera.polars as pa
import polars as pl

from mlops_core.config import DomainConfig, GroupSplit, ModelConfig
from mlops_core.contracts import check_contract
from mlops_core.ml.registry import ServedModel, load_champion
from mlops_core.storage import latest_partition, read_table, write_table

logger = logging.getLogger(__name__)


def predictions_schema(model: ModelConfig) -> pa.DataFrameSchema:
    """One row per scored item, carrying the item's keys."""
    items, split = model.items, model.training.split
    group = {split.column: pa.Column(pl.String)} if isinstance(split, GroupSplit) else {}
    return pa.DataFrameSchema(
        name=model.predictions_table,
        strict=True,
        unique=[items.id],
        columns={
            items.id: pa.Column(pl.String),
            items.period: pa.Column(pl.String),
            items.time: pa.Column(pl.Date),
            **group,
            "prediction": pa.Column(pl.Float64),
            # Which model produced the row: the join key for monitoring in stage 4.
            "model_version": pa.Column(pl.String),
            "predicted_at": pa.Column(pl.Datetime(time_unit="us", time_zone="UTC")),
        },
    )


def score(
    features: pl.DataFrame, served: ServedModel, model: ModelConfig, at: datetime
) -> pl.DataFrame:
    predictions = served.model.predict(features.select(model.spec.features).to_pandas())
    return features.select(model.keys).with_columns(
        pl.Series("prediction", predictions, dtype=pl.Float64),
        pl.lit(served.version).alias("model_version"),
        pl.lit(at).dt.replace_time_zone("UTC").alias("predicted_at"),
    )


def batch_predict(
    config: DomainConfig,
    model_name: str,
    data_dir: Path,
    tracking_uri: str,
    at: datetime | None = None,
) -> Path:
    """Score the named model's whole feature table with its champion."""
    model = config.model_named(model_name)
    registered = model.training.registered_model
    features_dir = data_dir / "features" / model.features_table
    features = read_table(features_dir)
    partition = latest_partition(features_dir)
    served = load_champion(registered, tracking_uri, data_dir / "model_cache")
    predicted_at = at or datetime.now(UTC)
    predictions = check_contract(
        predictions_schema(model), score(features, served, model, predicted_at)
    )
    logger.info(
        "Scored %d rows with %s v%s (%s)",
        predictions.height,
        registered,
        served.version,
        served.source,
    )
    return write_table(
        predictions,
        data_dir / "predictions" / model.predictions_table,
        {
            model.features_table: partition.name if partition else "",
            "model": f"{registered} v{served.version}",
        },
        predicted_at,
    )
