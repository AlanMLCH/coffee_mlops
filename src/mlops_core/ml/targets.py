"""Target transforms for experiments that ask a different question.

A raw score mixes two things: how good an item is *relative to its peers*, and the
level of the period it was graded in. The evaluation showed the second dominates the
error, so a within-period percentile is the right target when the question is "can these
features rank items at all?" — it removes the level by construction.

Predictions in percentile space are not interpretable to a buyer, so the inverse maps
them back to points using the **training** distribution. Using the test distribution
would leak exactly the level the experiment is trying to isolate.
"""

import numpy as np
import polars as pl


def percentile_within(frame: pl.DataFrame, target: str, group: str) -> pl.Series:
    """Rank each row against the rows of its own group, as a 0-1 percentile."""
    return (
        frame.select(
            (pl.col(target).rank("average").over(group) / pl.len().over(group)).alias("percentile")
        )
        .to_series()
        .cast(pl.Float64)
    )


def points_from_percentile(percentiles: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map percentiles back to points through the distribution of `reference`."""
    clipped = np.clip(percentiles, 0.0, 1.0)
    points: np.ndarray = np.quantile(reference, clipped)
    return points
