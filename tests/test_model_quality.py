"""Two ways a model is made better without being fitted to its test: a target judged by its
absolute error is learned as a median, and a range is calibrated on training rows its
edges never saw, so it holds the truth as often as it says."""

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError

from mlops_core.config import (
    GroupSplit,
    ItemsConfig,
    ModelConfig,
    ModelSpec,
    TargetBands,
    TemporalSplit,
    TrainingConfig,
)
from mlops_core.ml import train
from mlops_core.ml.band import Band
from mlops_core.ml.baseline_models import Lookup


def test_a_median_is_learned_for_a_quantity_only() -> None:
    median = ModelSpec(target="y", categorical=[], numeric=["a"], leakage=[], median=True)
    mean = ModelSpec(target="y", categorical=[], numeric=["a"], leakage=[])

    assert train.build_pipeline(median, {}, 0).named_steps["model"].objective == "regression_l1"
    assert train.build_pipeline(mean, {}, 0).named_steps["model"].objective == "regression"
    edge = train.build_pipeline(median, {}, 0, quantile=0.1).named_steps["model"]
    assert (edge.objective, edge.alpha) == ("quantile", 0.1)  # a range's edge stays a quantile
    with pytest.raises(ValidationError, match="median is learned for a quantity"):
        ModelSpec(target="y", categorical=[], numeric=["a"], leakage=[], median=True, task="count")


def model(split: TemporalSplit | GroupSplit) -> ModelConfig:
    return ModelConfig(
        name="toy",
        description="A toy range.",
        example={"a": 1.0},
        items=ItemsConfig(table="rows", id="item_id", time="day", period="period"),
        spec=ModelSpec(target="y", categorical=[], numeric=["a"], leakage=[], interval=0.8),
        training=TrainingConfig(
            split=split,
            cv_folds=2,
            trials=1,
            seed=0,
            baseline_group="period",
            bootstrap_resamples=100,
            min_probability_better=0.95,
            stratify_by="period",
            min_group_size=1,
            registered_model="toy",
        ),
        target_bands=TargetBands(edges=[1.0], labels=["low", "high"]),
    )


def rows(targets: list[float]) -> pl.DataFrame:
    n = len(targets)
    return pl.DataFrame(
        {
            "item_id": [f"i{i:03d}" for i in range(n)],
            "day": [date(2025, 1, 1) + timedelta(days=i) for i in range(n)],
            "period": ["p"] * n,
            "lot": [f"g{i % 10}" for i in range(n)],
            "a": [1.0] * n,
            "y": targets,
        }
    )


@pytest.fixture
def fixed_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """A band that always says [0, 1], whatever it is fitted on."""

    def band(spec: ModelSpec, params: dict[str, Any], seed: int) -> Band:
        return Band(Lookup(None, {}, 0.5), Lookup(None, {}, 0.0), Lookup(None, {}, 1.0))

    monkeypatch.setattr(train, "build_model", band)
    monkeypatch.setattr(train, "fit_params", lambda spec: {})


@pytest.mark.usefixtures("fixed_band")
def test_a_range_is_widened_by_how_far_the_latest_items_fell_outside_it() -> None:
    # Eighty items inside [0, 1], then the latest twenty from 1.1 to 3.0: past the top.
    late = [1.0 + 0.1 * k for k in range(1, 21)]
    temporal = TemporalSplit(kind="temporal", test_from=date(2026, 1, 1), recalibration_window=5)

    margin = train.conformal_margin(rows([0.5] * 80 + late), model(temporal), {}, 0)

    misses = np.array(late) - 1.0  # each one's distance above the top edge
    assert margin == pytest.approx(np.quantile(misses, 0.8 * (1 + 1 / 20)))
    assert train.conformal_margin(rows([0.5] * 20), model(temporal), {}, 0) == 0  # too few


@pytest.mark.usefixtures("fixed_band")
def test_a_range_too_wide_is_never_narrowed_and_whole_groups_calibrate_a_group_model() -> None:
    """Every held-out item sits half a unit inside each edge: the misses' quantile is -0.5,
    but a range narrowed on a calm past fails when the future is not calm."""
    grouped = GroupSplit(kind="group", column="lot", test_share=0.25)

    assert train.conformal_margin(rows([0.5] * 100), model(grouped), {}, 0) == 0
    late = [0.5] * 80 + [3.0] * 20  # the held-out groups' items fall above the band
    assert train.conformal_margin(rows(late), model(grouped), {}, 0) > 0


def test_a_band_moves_its_edges_by_its_margin_and_an_old_one_by_none() -> None:
    x = pd.DataFrame({"a": [1.0]})
    band = Band(Lookup(None, {}, 0.5), Lookup(None, {}, 0.0), Lookup(None, {}, 1.0), margin=0.3)
    old = Band(Lookup(None, {}, 0.5), Lookup(None, {}, 0.0), Lookup(None, {}, 1.0))
    del old.margin  # as a range saved before margins existed loads
    narrowed = Band(
        Lookup(None, {}, 0.5), Lookup(None, {}, 0.4), Lookup(None, {}, 0.6), margin=-0.2
    )

    assert [edge.tolist() for edge in band.band(x)] == [[-0.3], [1.3]]
    assert [edge.tolist() for edge in old.band(x)] == [[0.0], [1.0]]
    # Narrowed past its own centre, a range is its two edges in order, never upside down.
    low, high = narrowed.band(x)
    assert (low.tolist(), high.tolist()) == (pytest.approx([0.4]), pytest.approx([0.6]))
