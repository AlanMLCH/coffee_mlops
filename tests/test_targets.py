import numpy as np
import polars as pl
import pytest

from mlops_core.ml.targets import percentile_within, points_from_percentile


def test_the_percentile_is_computed_inside_each_period() -> None:
    """Both periods contain the same ranking at different levels; the percentiles match."""
    frame = pl.DataFrame(
        {
            "snapshot": ["old"] * 3 + ["new"] * 3,
            "points": [80.0, 82.0, 84.0, 84.0, 86.0, 88.0],
        }
    )

    percentiles = percentile_within(frame, "points", "snapshot")

    assert percentiles.to_list() == pytest.approx([1 / 3, 2 / 3, 1.0] * 2)


def test_ties_share_their_rank() -> None:
    frame = pl.DataFrame({"g": ["a"] * 4, "points": [80.0, 82.0, 82.0, 84.0]})

    percentiles = percentile_within(frame, "points", "g")

    assert percentiles.to_list() == pytest.approx([0.25, 0.625, 0.625, 1.0])


def test_percentiles_map_back_through_the_training_distribution() -> None:
    reference = np.array([80.0, 82.0, 84.0, 86.0, 88.0])

    points = points_from_percentile(np.array([0.0, 0.5, 1.0]), reference)

    assert points.tolist() == [80.0, 84.0, 88.0]


def test_predictions_outside_the_range_are_clipped_not_extrapolated() -> None:
    reference = np.array([80.0, 90.0])

    points = points_from_percentile(np.array([-0.4, 1.7]), reference)

    assert points.tolist() == [80.0, 90.0]
