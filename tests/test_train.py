from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from coffee_mlops.clean import build_clean
from coffee_mlops.config import DomainConfig
from coffee_mlops.features import build_features
from coffee_mlops.train import (
    CHAMPION,
    baseline_predictions,
    build_pipeline,
    fit_params,
    promote_if_better,
    temporal_split,
    train_model,
    xy,
)


@pytest.fixture
def fast_config(coffee_config: DomainConfig) -> DomainConfig:
    """Same config, but a tuning budget small enough for a unit test."""
    training = coffee_config.training.model_copy(update={"trials": 2, "cv_folds": 2})
    return coffee_config.model_copy(update={"training": training})


@pytest.fixture
def data_dir(fast_config: DomainConfig, raw_dir: Path) -> Path:
    build_clean(fast_config, raw_dir.parent)
    build_features(fast_config, raw_dir.parent)
    return raw_dir.parent


def features_frame(countries: list[str | None], points: list[float]) -> pl.DataFrame:
    n = len(countries)
    return pl.DataFrame(
        {
            "grading_date": [date(2015, 1, 1 + i) for i in range(n)],
            "country": countries,
            "variety": ["caturra"] * n,
            "processing_method": ["washed"] * n,
            "color": ["green"] * n,
            **{
                c: [1.0] * n
                for c in (
                    "altitude_m",
                    "moisture_pct",
                    "category_one_defects",
                    "category_two_defects",
                    "quakers",
                    "ctx_production",
                    "ctx_arabica_share",
                    "ctx_export_share",
                    "ctx_domestic_consumption",
                )
            },
            "total_cup_points": points,
        }
    )


def test_temporal_split_never_trains_on_the_future(fast_config: DomainConfig) -> None:
    features = pl.DataFrame(
        {"grading_date": [date(2023, 1, 1), date(2010, 1, 1), date(2018, 12, 31), date(2019, 1, 1)]}
    )

    train, test = temporal_split(features, fast_config.training)

    assert train["grading_date"].to_list() == [date(2010, 1, 1), date(2018, 12, 31)]
    assert test["grading_date"].to_list() == [date(2019, 1, 1), date(2023, 1, 1)]


def test_group_baseline_falls_back_to_global_mean_for_unseen_groups(
    fast_config: DomainConfig,
) -> None:
    train = features_frame(["Mexico", "Mexico", "Kenya"], [80.0, 82.0, 84.0])
    test = features_frame(["Kenya", "Laos"], [85.0, 84.0])

    baselines = baseline_predictions(train, test, fast_config.model, "country")

    assert baselines["global_mean"].tolist() == [82.0, 82.0]
    assert baselines["country_mean"].tolist() == [84.0, 82.0]


def test_pipeline_serves_unseen_and_missing_categories(fast_config: DomainConfig) -> None:
    spec = fast_config.model
    train = features_frame(["Mexico", "Kenya"] * 10, [80.0, 86.0] * 10)
    pipeline = build_pipeline(spec, {"n_estimators": 10, "min_child_samples": 2}, seed=0)
    pipeline.fit(*xy(train, spec), **fit_params(spec))

    unseen = features_frame(["Narnia", None], [0.0, 0.0])
    predictions = pipeline.predict(xy(unseen, spec)[0])

    assert np.isfinite(predictions).all()


@dataclass
class FakeRegistry:
    """Just the registry calls the promotion gate makes."""

    champion_mae: float | None = None
    aliases: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        if self.champion_mae is None:
            raise MlflowException("no alias")
        return type("Version", (), {"run_id": "champion-run"})()

    def get_run(self, run_id: str) -> Any:
        metrics = {"test_mae": self.champion_mae}
        return type("Run", (), {"data": type("Data", (), {"metrics": metrics})()})()

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags[version] = value


@pytest.mark.parametrize(
    ("champion_mae", "test_mae", "promoted"),
    [
        pytest.param(None, 1.9, False, id="not-better-than-baseline"),
        pytest.param(None, 1.5, True, id="first-model-beating-baseline"),
        pytest.param(1.4, 1.5, False, id="worse-than-champion"),
        pytest.param(1.6, 1.5, True, id="better-than-champion"),
    ],
)
def test_quality_gate(champion_mae: float | None, test_mae: float, promoted: bool) -> None:
    registry = FakeRegistry(champion_mae=champion_mae)

    result = promote_if_better(registry, "m", "7", test_mae=test_mae, gate_mae=1.8)  # type: ignore[arg-type]

    assert result is promoted
    assert (registry.aliases.get(CHAMPION) == "7") is promoted
    assert registry.tags["7"].startswith("promoted" if promoted else "rejected")


def test_training_is_tracked_registered_and_servable(
    fast_config: DomainConfig,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)  # MLflow writes local artifacts under the working dir
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    result = train_model(fast_config, data_dir, tracking_uri)

    client = MlflowClient(tracking_uri)
    run = client.get_run(result.run_id)
    assert {"cv_mae", "test_mae", "test_bias", "baseline_country_mean_test_mae"} <= set(
        run.data.metrics
    )
    assert run.data.tags["features_partition"].startswith("built_at=")
    trials = client.search_runs(
        run.info.experiment_id, f"tags.mlflow.parentRunId = '{result.run_id}'"
    )
    assert len(trials) == 2

    name = fast_config.training.registered_model
    model = mlflow.sklearn.load_model(f"models:/{name}/{result.model_version}")
    x_test = xy(
        temporal_split(
            pl.read_parquet(next(data_dir.rglob("review_features.parquet"))), fast_config.training
        )[1],
        fast_config.model,
    )[0]
    assert len(model.predict(x_test)) == 12
    if result.promoted:
        assert client.get_model_version_by_alias(name, CHAMPION).version == result.model_version
