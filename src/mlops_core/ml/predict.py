"""Batch inference.

Scores the whole feature table with the champion and writes the result as another
immutable Parquet partition, so predictions are queryable next to the data that
produced them and a running API or agent is never blocked by the job.

A model with a range writes its edges too. A model split by group also writes what each
item's prediction would be from a model that never saw its group: a residual read off
the champion's own prediction is shrunk wherever it learned the item, so ranking items
by how far they sit from what they should be needs predictions that did not see them.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandera.polars as pa
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from mlops_core.config import DomainConfig, GroupSplit, ModelConfig
from mlops_core.contracts import check_contract
from mlops_core.ml.band import predicted
from mlops_core.ml.registry import ServedModel, load_champion
from mlops_core.ml.train import fit_params
from mlops_core.storage import latest_partition, read_table, write_table

HELD_OUT = "held_out_prediction"

logger = logging.getLogger(__name__)


def predictions_schema(model: ModelConfig) -> pa.DataFrameSchema:
    """One row per scored item, carrying the item's keys."""
    items, split = model.items, model.training.split
    group = {split.column: pa.Column(pl.String)} if isinstance(split, GroupSplit) else {}
    entity = {items.entity: pa.Column(pl.String)} if items.entity else {}
    band = (
        {"lower": pa.Column(pl.Float64), "upper": pa.Column(pl.Float64)}
        if model.spec.interval is not None
        else {}
    )
    held_out = {HELD_OUT: pa.Column(pl.Float64)} if isinstance(split, GroupSplit) else {}
    return pa.DataFrameSchema(
        name=model.predictions_table,
        strict=True,
        unique=[items.id],
        columns={
            items.id: pa.Column(pl.String),
            items.period: pa.Column(pl.String),
            items.time: pa.Column(pl.Date),
            **entity,
            **group,
            "prediction": pa.Column(pl.Float64),
            **band,
            **held_out,
            # Which model produced the row: the join key for monitoring in stage 4.
            "model_version": pa.Column(pl.String),
            "predicted_at": pa.Column(pl.Datetime(time_unit="us", time_zone="UTC")),
        },
    )


def score(
    features: pl.DataFrame, served: ServedModel, model: ModelConfig, at: datetime
) -> pl.DataFrame:
    scored = predicted(served.model, features.select(model.spec.features).to_pandas())
    band = (
        [
            pl.Series("lower", scored.lower, dtype=pl.Float64),
            pl.Series("upper", scored.upper, dtype=pl.Float64),
        ]
        if scored.lower is not None and scored.upper is not None
        else []
    )
    held_out = (
        [pl.Series(HELD_OUT, held_out_predictions(features, served, model), dtype=pl.Float64)]
        if isinstance(model.training.split, GroupSplit)
        else []
    )
    return features.select(model.keys).with_columns(
        pl.Series("prediction", scored.point, dtype=pl.Float64),
        *band,
        *held_out,
        pl.lit(served.version).alias("model_version"),
        pl.lit(at).dt.replace_time_zone("UTC").alias("predicted_at"),
    )


def held_out_predictions(
    features: pl.DataFrame, served: ServedModel, model: ModelConfig
) -> np.ndarray:
    """Each item as predicted by the champion's own recipe - its hyperparameters,
    refitted - on the other groups' items, fold by fold. An item without a target was
    never learned from: it keeps the champion's prediction."""
    spec, cfg, split = model.spec, model.training, model.training.split
    if not isinstance(split, GroupSplit):
        raise TypeError(f"{model.name} is not split by group")
    x = features.select(spec.features).to_pandas()
    held_out: np.ndarray = np.asarray(served.model.predict(x), dtype=float)
    known = np.flatnonzero(features[spec.target].is_not_null().to_numpy())
    y, groups = features[spec.target].to_numpy(), features[split.column].to_numpy()
    for fit_rows, held_rows in GroupKFold(n_splits=cfg.cv_folds).split(known, groups=groups[known]):
        refitted = clone(served.model).fit(
            x.iloc[known[fit_rows]], y[known[fit_rows]], **fit_params(spec)
        )
        held_out[known[held_rows]] = refitted.predict(x.iloc[known[held_rows]])
    return held_out


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
