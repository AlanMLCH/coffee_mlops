"""Evaluation shared by training and analysis: errors, metrics by group, and the
recalibration a deployed model would get. The paired bootstrap that turns two sets of
errors into a gate decision is in `mlops_core.stats`, shared with retrieval.
"""

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from mlops_core.config import ModelSpec
from mlops_core.stats import bootstrap_means


def absolute_errors(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-row absolute error, the unit every paired comparison here works on."""
    errors: np.ndarray = np.abs(prediction - y)
    return errors


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(root_mean_squared_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
        # Mean over- (+) or under- (-) prediction: the level shift the model cannot see.
        "bias": float(np.mean(prediction - y)),
    }


def mae_interval(
    errors: np.ndarray, resamples: int = 5000, seed: int = 0, groups: np.ndarray | None = None
) -> tuple[float, float]:
    """95% interval for the MAE itself: how precise the headline number is."""
    means = bootstrap_means(errors, resamples, seed, groups)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stratified_metrics(
    train: pl.DataFrame,
    test: pl.DataFrame,
    prediction: np.ndarray,
    spec: ModelSpec,
    group: str,
    min_group_size: int,
) -> pl.DataFrame:
    """Per-group error **and** how the group's weight changed between the splits.

    A temporal split rarely shifts time alone: if a country goes from 6% of training to
    30% of test, an overall metric mixes drift with a different population. The share
    columns make that visible instead of leaving it as an unexplained error.
    """
    scored = test.with_columns(
        pl.Series("_prediction", prediction),
        (pl.Series("_prediction", prediction) - pl.col(spec.target)).alias("_error"),
    )
    train_shares = (
        train.group_by(group)
        .len()
        .with_columns((pl.col("len") / train.height).alias("train_share"))
    )
    return (
        scored.group_by(group)
        .agg(
            pl.len().alias("n_test"),
            (pl.len() / scored.height).alias("test_share"),
            pl.col("_error").abs().mean().alias("mae"),
            pl.col("_error").mean().alias("bias"),
            pl.col(spec.target).mean().alias("observed"),
        )
        .join(train_shares.select(group, "train_share"), on=group, how="left")
        .with_columns(pl.col("train_share").fill_null(0.0))
        .filter(pl.col("n_test") >= min_group_size)
        .sort("mae", descending=True)
    )


def recalibration_gain(
    test: pl.DataFrame, prediction: np.ndarray, spec: ModelSpec, window: int, time: str
) -> dict[str, float]:
    """What a deployed recalibration would buy, measured honestly.

    Simulates what a monitoring loop does: take the first `window` items of the new
    period (in `time` order), estimate the level shift from them alone, and apply that offset to
    everything after. Both metrics are computed on the rows *after* the window, so the
    offset is never estimated on the rows it is scored against.
    """
    ordered = test.with_columns(pl.Series("_prediction", prediction)).sort(time)
    if ordered.height <= window:
        return {}
    observed = ordered[spec.target].to_numpy()
    predicted = ordered["_prediction"].to_numpy()
    offset = float(np.mean(predicted[:window] - observed[:window]))
    held_out = slice(window, None)
    return {
        "recalibration_offset": offset,
        "recalibration_n_holdout": float(ordered.height - window),
        "mae_before_recalibration": float(np.abs(predicted[held_out] - observed[held_out]).mean()),
        "mae_after_recalibration": float(
            np.abs(predicted[held_out] - offset - observed[held_out]).mean()
        ),
    }
