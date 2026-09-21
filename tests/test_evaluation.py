from datetime import date

import numpy as np
import polars as pl
import pytest

from mlops_core.config import DomainConfig
from mlops_core.ml.evaluation import (
    absolute_errors,
    compare,
    mae_interval,
    recalibration_gain,
    stratified_metrics,
)

RESAMPLES = 400  # enough to be stable, small enough to stay fast


def test_a_consistently_better_candidate_is_reported_as_certain() -> None:
    candidate, reference = np.full(50, 1.0), np.full(50, 2.0)

    result = compare(candidate, reference, RESAMPLES, seed=0)

    assert result.difference == -1.0
    assert result.probability_better == 1.0
    assert result.ci_high < 0


def test_identical_models_are_a_coin_flip() -> None:
    errors = np.array([1.0, 2.0, 3.0, 0.5] * 10)

    result = compare(errors, errors.copy(), RESAMPLES, seed=0)

    assert result.difference == 0.0
    assert result.ci_low == result.ci_high == 0.0
    assert result.probability_better == 0.0  # never strictly better


def test_a_tiny_edge_on_few_rows_is_not_certain() -> None:
    """The case the gate exists for: better on average, but well inside the noise."""
    rng = np.random.default_rng(0)
    reference = rng.normal(2.0, 1.0, 40)
    candidate = reference + rng.normal(-0.02, 1.0, 40)

    result = compare(candidate, reference, RESAMPLES, seed=1)

    assert result.probability_better < 0.95


def test_mae_interval_brackets_the_point_estimate() -> None:
    errors = absolute_errors(np.array([80.0, 82.0, 84.0]), np.array([81.0, 80.0, 85.0]))

    low, high = mae_interval(errors, RESAMPLES, seed=0)

    assert low <= errors.mean() <= high


def frame(
    countries: list[str], points: list[float], start: date = date(2023, 1, 1)
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country": countries,
            "total_cup_points": points,
            "grading_date": [date(start.year, start.month, 1 + i) for i in range(len(points))],
        }
    )


def test_stratified_metrics_expose_a_composition_shift(coffee_config: DomainConfig) -> None:
    train = frame(["Mexico"] * 9 + ["Taiwan"], [82.0] * 10)
    test = frame(["Taiwan"] * 6 + ["Mexico"] * 4, [84.0] * 10)
    prediction = np.full(10, 82.0)

    metrics = stratified_metrics(train, test, prediction, coffee_config.model, "country", 3)

    taiwan = metrics.filter(pl.col("country") == "Taiwan").row(0, named=True)
    assert taiwan["train_share"] == pytest.approx(0.1)
    assert taiwan["test_share"] == pytest.approx(0.6)
    assert taiwan["mae"] == 2.0
    assert taiwan["bias"] == -2.0  # the model under-predicts the new population


def test_small_groups_are_left_out(coffee_config: DomainConfig) -> None:
    train = frame(["Mexico"] * 5, [82.0] * 5)
    test = frame(["Mexico"] * 5 + ["Laos"], [82.0] * 6)

    metrics = stratified_metrics(
        train, test, np.full(6, 82.0), coffee_config.model, "country", min_group_size=3
    )

    assert metrics["country"].to_list() == ["Mexico"]


def test_recalibration_is_measured_on_rows_it_never_saw(coffee_config: DomainConfig) -> None:
    # Every prediction is 1.5 points low: a pure level shift, like the 2023 snapshot.
    test = frame(["Mexico"] * 20, [84.0] * 20)
    prediction = np.full(20, 82.5)

    gain = recalibration_gain(test, prediction, coffee_config.model, window=5)

    assert gain["recalibration_offset"] == pytest.approx(-1.5)
    assert gain["recalibration_n_holdout"] == 15
    assert gain["mae_before_recalibration"] == pytest.approx(1.5)
    assert gain["mae_after_recalibration"] == pytest.approx(0.0, abs=1e-9)


def test_recalibration_needs_more_rows_than_the_window(coffee_config: DomainConfig) -> None:
    test = frame(["Mexico"] * 5, [84.0] * 5)

    assert recalibration_gain(test, np.full(5, 82.5), coffee_config.model, window=30) == {}
