"""Evaluation shared by training and analysis: errors, metrics by group, and the
recalibration a deployed model would get. The paired bootstrap that turns two sets of
errors into a gate decision is in `mlops_core.stats`, shared with retrieval.

Each row's loss depends on what the model predicts (`ModelSpec.task`): an absolute
error for a quantity, a Poisson deviance for a count, a Brier score for a probability,
an interval score for a range. Whatever it is, lower is better and it is paired row by
row, so one gate serves them all.
"""

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error

from mlops_core.config import ModelSpec
from mlops_core.ml.band import Predicted
from mlops_core.stats import bootstrap_means

# A predicted rate of zero makes any observed count infinitely surprising: rates are
# floored here, so a baseline that predicts zero is judged very badly instead of not at all.
SMALLEST_RATE = 1e-6
# A probability of exactly 0 or 1 that turns out wrong has an infinite log loss.
SMALLEST_PROBABILITY = 1e-6


def absolute_errors(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-row absolute error, the unit every paired comparison here works on."""
    errors: np.ndarray = np.abs(prediction - y)
    return errors


def loss_name(spec: ModelSpec) -> str:
    """What the gate compares for this model, as metrics name it."""
    if spec.interval is not None:
        return "interval_score"
    return {"regression": "mae", "count": "poisson_deviance", "probability": "brier"}[spec.task]


def row_losses(spec: ModelSpec, y: np.ndarray, scored: Predicted) -> np.ndarray:
    """Each row's loss, the unit the gate pairs: lower is better, whatever the task."""
    if spec.interval is not None:
        if scored.lower is None or scored.upper is None:
            raise ValueError("A range is judged by its range: this prediction has none")
        return interval_scores(y, scored.lower, scored.upper, spec.interval)
    if spec.task == "count":
        return poisson_deviances(y, scored.point)
    if spec.task == "probability":
        brier: np.ndarray = (scored.point - y) ** 2
        return brier
    return absolute_errors(y, scored.point)


def poisson_deviances(y: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Per-row Poisson deviance: how surprising each count is at the predicted rate."""
    rate = np.maximum(rate, SMALLEST_RATE)
    ratio = np.where(y > 0, y / rate, 1.0)  # a zero count contributes no log term
    deviance: np.ndarray = 2 * (y * np.log(ratio) - (y - rate))
    return deviance


def interval_scores(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, coverage: float
) -> np.ndarray:
    """Per-row interval score (Gneiting and Raftery, 2007): the range's width, plus a
    penalty for a miss that grows with the distance. A range can win only by being narrow
    *and* right as often as it claims: a wide one always covers and pays for its width, a
    narrow one misses and pays more for each miss than the width it saved."""
    penalty = 2 / (1 - coverage)
    scores: np.ndarray = (
        (upper - lower)
        + penalty * np.maximum(lower - y, 0.0)
        + penalty * np.maximum(y - upper, 0.0)
    )
    return scores


def task_metrics(spec: ModelSpec, y: np.ndarray, scored: Predicted) -> dict[str, float]:
    """The regression metrics of the point, and what the task adds: a count's deviance,
    a probability's calibration and ranking, a range's coverage and width."""
    metrics = regression_metrics(y, scored.point)
    if spec.task == "count":
        metrics["poisson_deviance"] = float(poisson_deviances(y, scored.point).mean())
    if spec.task == "probability":
        p = np.clip(scored.point, SMALLEST_PROBABILITY, 1 - SMALLEST_PROBABILITY)
        metrics |= {
            "brier": float(((scored.point - y) ** 2).mean()),
            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "base_rate": float(y.mean()),
            "mean_probability": float(scored.point.mean()),
        }
        if 0 < y.sum() < len(y):  # ranking needs a yes and a no to rank
            metrics["auc"] = float(roc_auc_score(y, scored.point))
    if spec.interval is not None and scored.lower is not None and scored.upper is not None:
        metrics |= {
            "coverage": float(((scored.lower <= y) & (y <= scored.upper)).mean()),
            "mean_width": float((scored.upper - scored.lower).mean()),
            "interval_score": float(
                interval_scores(y, scored.lower, scored.upper, spec.interval).mean()
            ),
        }
    return metrics


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
    prediction: np.ndarray | Predicted,
    spec: ModelSpec,
    group: str,
    min_group_size: int,
) -> pl.DataFrame:
    """Per-group error **and** how the group's weight changed between the splits.

    A temporal split rarely shifts time alone: if a country goes from 6% of training to
    30% of test, an overall metric mixes drift with a different population. The share
    columns make that visible instead of leaving it as an unexplained error. A range adds
    how often it held the truth in each group: a range right four times in five overall
    can be right always at one horizon and half the time at another.
    """
    scored_rows = prediction if isinstance(prediction, Predicted) else Predicted(prediction)
    point = scored_rows.point
    scored = test.with_columns(
        pl.Series("_prediction", point),
        (pl.Series("_prediction", point) - pl.col(spec.target)).alias("_error"),
    )
    held = []
    if scored_rows.lower is not None and scored_rows.upper is not None:
        y = test[spec.target].to_numpy()
        inside = (scored_rows.lower <= y) & (y <= scored_rows.upper)
        scored = scored.with_columns(
            pl.Series("_inside", inside.astype(float)),
            pl.Series("_width", scored_rows.upper - scored_rows.lower),
        )
        held = [pl.col("_inside").mean().alias("coverage"), pl.col("_width").mean().alias("width")]
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
            *held,
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
