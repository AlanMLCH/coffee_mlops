"""Evaluation shared by training and analysis.

With a test split of a couple of hundred rows, point metrics decide nothing: a gap of
0.1 MAE sits inside the noise. So comparisons here are **paired** (same rows, both
models) and bootstrapped, which lets a quality gate ask "how sure are we?" instead of
"which number is bigger?". A paired comparison is far more sensitive than comparing two
independent confidence intervals, because the rows a model finds hard are hard for both.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from mlops_core.config import ModelSpec


@dataclass(frozen=True)
class Comparison:
    """How much better a candidate is than a reference, measured on the same rows."""

    # Mean paired difference of absolute errors; negative means the candidate is better.
    difference: float
    ci_low: float
    ci_high: float
    probability_better: float

    def as_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_difference": self.difference,
            f"{prefix}_ci_low": self.ci_low,
            f"{prefix}_ci_high": self.ci_high,
            f"{prefix}_probability_better": self.probability_better,
        }


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


def _bootstrap_means(values: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """Means of `resamples` resamples drawn with replacement, all at once."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    return values[draws].mean(axis=1)


def mae_interval(errors: np.ndarray, resamples: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the MAE itself: how precise the headline number is."""
    means = _bootstrap_means(errors, resamples, seed)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(
    candidate: np.ndarray, reference: np.ndarray, resamples: int = 5000, seed: int = 0
) -> Comparison:
    """Paired bootstrap of `candidate - reference` absolute errors, row by row."""
    difference = candidate - reference
    means = _bootstrap_means(difference, resamples, seed)
    return Comparison(
        difference=float(difference.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        # The candidate wins in this share of resamples (lower error = negative mean).
        probability_better=float((means < 0).mean()),
    )


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
    test: pl.DataFrame, prediction: np.ndarray, spec: ModelSpec, window: int
) -> dict[str, float]:
    """What a deployed recalibration would buy, measured honestly.

    Simulates what a monitoring loop does: take the first `window` graded lots of the
    new period, estimate the level shift from them alone, and apply that offset to
    everything after. Both metrics are computed on the rows *after* the window, so the
    offset is never estimated on the rows it is scored against.
    """
    ordered = test.with_columns(pl.Series("_prediction", prediction)).sort("grading_date")
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
