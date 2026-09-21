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

from mlops_core.config import DomainConfig, ModelSpec
from mlops_core.contracts import check_contract
from mlops_core.ml.registry import ServedModel, load_champion
from mlops_core.storage import latest_partition, read_table, write_table

logger = logging.getLogger(__name__)

FEATURES_TABLE = "review_features"
PREDICTIONS_TABLE = "review_predictions"

PREDICTIONS = pa.DataFrameSchema(
    name=PREDICTIONS_TABLE,
    strict=True,
    unique=["review_id"],
    columns={
        "review_id": pa.Column(pl.String),
        "snapshot": pa.Column(pl.String),
        "grading_date": pa.Column(pl.Date),
        "prediction": pa.Column(pl.Float64),
        # Which model produced the row: the join key for monitoring in stage 4.
        "model_version": pa.Column(pl.String),
        "predicted_at": pa.Column(pl.Datetime(time_unit="us", time_zone="UTC")),
    },
)


def score(
    features: pl.DataFrame, served: ServedModel, spec: ModelSpec, at: datetime
) -> pl.DataFrame:
    predictions = served.model.predict(features.select(spec.features).to_pandas())
    return features.select("review_id", "snapshot", "grading_date").with_columns(
        pl.Series("prediction", predictions, dtype=pl.Float64),
        pl.lit(served.version).alias("model_version"),
        pl.lit(at).dt.replace_time_zone("UTC").alias("predicted_at"),
    )


def batch_predict(
    config: DomainConfig, data_dir: Path, tracking_uri: str, at: datetime | None = None
) -> Path:
    features_dir = data_dir / "features" / FEATURES_TABLE
    features = read_table(features_dir)
    partition = latest_partition(features_dir)
    served = load_champion(config.training.registered_model, tracking_uri, data_dir / "model_cache")
    predicted_at = at or datetime.now(UTC)
    predictions = check_contract(PREDICTIONS, score(features, served, config.model, predicted_at))
    logger.info(
        "Scored %d rows with %s v%s (%s)",
        predictions.height,
        config.training.registered_model,
        served.version,
        served.source,
    )
    return write_table(
        predictions,
        data_dir / "predictions" / PREDICTIONS_TABLE,
        {
            FEATURES_TABLE: partition.name if partition else "",
            "model": f"{config.training.registered_model} v{served.version}",
        },
        predicted_at,
    )
