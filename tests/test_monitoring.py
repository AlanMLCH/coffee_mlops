"""The monitor: which period is compared with which, what calls for retraining, and the
record it leaves.

Periods are written here with a known shift in one column and none in the others, so
what Evidently finds is what the test put there.
"""

import logging
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow import MlflowClient

from mlops_core.config import DomainConfig, ModelConfig
from mlops_core.monitoring.drift import (
    MONITORING,
    REPORT_FILE,
    accepted_error,
    detect_drift,
    monitor_model,
)
from mlops_core.storage import read_table, write_table

ROWS = 200  # per period: enough for a shift of one standard deviation to be found


def model() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "name": "price",
            "description": "A price.",
            "example": {"origin": "a"},
            "items": {"table": "lots", "id": "item_id", "time": "observed_on", "period": "year"},
            "spec": {"target": "price", "categorical": ["origin"], "numeric": ["size", "age"],
                     "leakage": []},
            "training": {
                "split": {"kind": "temporal", "test_from": "2026-01-01", "recalibration_window": 5},
                "cv_folds": 2, "trials": 1, "seed": 0, "baseline_group": "origin",
                "bootstrap_resamples": 100, "min_probability_better": 0.95,
                "stratify_by": "origin", "min_group_size": 1, "registered_model": "toy-price",
            },
            "target_bands": {"edges": [100.0], "labels": ["cheap", "dear"]},
        }
    )  # fmt: skip


def features(shift: dict[str, float] | None = None, seed: int = 0) -> pl.DataFrame:
    """Three yearly periods; `shift` moves a numeric column in the last one."""
    rng = np.random.default_rng(seed)
    frames = []
    for year in (2024, 2025, 2026):
        moved = shift if year == 2026 and shift else {}
        frames.append(
            pl.DataFrame(
                {
                    "item_id": [f"{year}-{n}" for n in range(ROWS)],
                    "year": [str(year)] * ROWS,
                    "observed_on": [date(year, 1, 1)] * ROWS,
                    "origin": rng.choice(["a", "b"], ROWS).tolist(),
                    **{
                        column: (rng.normal(100, 10, ROWS) + moved.get(column, 0) * 10).tolist()
                        for column in ("size", "age", "price")
                    },
                }
            )
        )
    return pl.concat(frames)


def predictions(frame: pl.DataFrame, error: float) -> pl.DataFrame:
    return frame.select("item_id", (pl.col("price") + error).alias("prediction"))


def test_the_newest_period_is_compared_with_every_one_before_it() -> None:
    result = detect_drift(model(), features({"size": 1.0}), None, 0.5, None)

    assert result is not None
    assert (result.reference, result.current) == (["2024", "2025"], "2026")
    drifted = dict(result.columns.select("column", "drifted").iter_rows())
    assert drifted == {"size": True, "age": False, "price": False, "origin": False}
    assert result.drifted_share == pytest.approx(1 / 3)
    roles = dict(result.columns.select("column", "role").iter_rows())
    assert roles == {"size": "feature", "age": "feature", "price": "target", "origin": "feature"}
    assert not result.retrain  # one feature in three: below the line, target unmoved
    assert "<html" in result.report_html.lower()


def test_many_drifted_features_or_a_drifted_target_call_for_retraining() -> None:
    result = detect_drift(
        model(), features({"size": 1.0, "age": 1.0, "price": 1.0}), None, 0.5, None
    )

    assert result is not None
    assert result.reasons == [
        "67% of the features drifted (the line is 50%)",
        "the target, price, drifted",
    ]


def test_an_error_above_the_accepted_interval_calls_for_retraining() -> None:
    table = features()
    scored = predictions(table, error=2.0)

    within = detect_drift(model(), table, scored, 0.5, accepted_mae=2.5)
    beyond = detect_drift(model(), table, scored, 0.5, accepted_mae=1.5)
    unknown = detect_drift(model(), table, scored, 0.5, accepted_mae=None)

    assert within is not None and beyond is not None and unknown is not None
    assert within.current_mae == pytest.approx(2.0) and within.reasons == []
    assert beyond.reasons == [
        "the error on 2026 is 2.000, above the 1.500 the champion was accepted with"
    ]
    assert unknown.reasons == []  # nothing to hold it to
    assert "prediction" in beyond.columns["column"].to_list()
    unscored = scored.filter(~pl.col("item_id").str.starts_with("2026"))
    not_yet = detect_drift(model(), table, unscored, 0.5, accepted_mae=1.5)
    assert not_yet is not None and not_yet.current_mae is None  # the newest period unscored


def test_one_period_is_nothing_to_compare_and_an_empty_column_is_not_tested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = features()

    assert detect_drift(model(), table.filter(pl.col("year") == "2026"), None, 0.5, None) is None
    with caplog.at_level(logging.INFO):
        blank = table.with_columns(
            pl.when(pl.col("year") == "2026").then(None).otherwise(pl.col("age")).alias("age")
        )
        result = detect_drift(model(), blank, None, 0.5, None)
    assert result is not None and "age" not in result.columns["column"].to_list()
    assert "not tested: ['age']" in caplog.text


def champion(name: str, metrics: dict[str, float]) -> None:
    """A registered model whose champion's run logged `metrics`."""
    mlflow.set_experiment("training")  # not whichever an earlier test left active
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
    client = MlflowClient()
    client.create_registered_model(name)
    version = client.create_model_version(name, f"runs:/{run.info.run_id}/model", run.info.run_id)
    client.set_registered_model_alias(name, "champion", version.version)


def test_the_accepted_error_is_the_champions_interval(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    champion("with-interval", {"test_mae": 1.6, "test_mae_ci_high": 1.9})
    champion("without-interval", {"test_mae": 1.6})

    assert accepted_error("with-interval") == 1.9
    assert accepted_error("never-registered") is None
    with caplog.at_level(logging.WARNING):
        assert accepted_error("without-interval") is None
    assert "logged without a test MAE interval" in caplog.text


def test_the_monitor_writes_its_table_and_report_and_logs_a_run(
    tmp_path: Path, coffee_config: DomainConfig
) -> None:
    config = coffee_config.model_copy(update={"models": [model()]})
    data_dir = tmp_path / "coffee"
    table = features({"size": 1.0, "age": 1.0})
    write_table(table, data_dir / "features" / "price_features", {})
    write_table(predictions(table, 3.0), data_dir / "predictions" / "price_predictions", {})
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    champion("toy-price", {"test_mae_ci_high": 2.0})

    result = monitor_model(config, "price", data_dir, uri)

    assert result is not None and result.retrain
    written = read_table(data_dir / MONITORING / "price_drift")
    assert written.equals(result.columns)
    partition = next((data_dir / MONITORING / "price_drift").glob("built_at=*"))
    assert "<!doctype html>" in (partition / REPORT_FILE).read_text(encoding="utf-8").lower()
    (run,) = mlflow.search_runs(experiment_names=["coffee-monitoring"], output_format="list")
    assert run.data.tags["retrain"] == "True"
    assert run.data.params["current_period"] == "2026"
    assert run.data.metrics["current_mae"] == pytest.approx(3.0)
    assert run.data.metrics["accepted_mae"] == 2.0


def test_a_model_with_one_period_leaves_no_record(
    tmp_path: Path, coffee_config: DomainConfig
) -> None:
    config = coffee_config.model_copy(update={"models": [model()]})
    data_dir = tmp_path / "coffee"
    write_table(
        features().filter(pl.col("year") == "2026"), data_dir / "features" / "price_features", {}
    )

    assert monitor_model(config, "price", data_dir, f"sqlite:///{tmp_path.as_posix()}/m.db") is None
    assert not (data_dir / MONITORING).exists()
