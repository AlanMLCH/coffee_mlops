"""Drift, per model: the newest period against every one before it.

A model's items come in periods (its config names the column: a snapshot, a catalogue
read, a decade). The newest period is compared with all the earlier ones, column by
column - every feature the model reads, its target, and its champion's predictions - with
Evidently's statistical tests, which it picks by column type and sample size (K-S or
Wasserstein for numbers, chi-squared, Z or Jensen-Shannon for categories). A column has
drifted when its test says so at Evidently's own threshold.

A model is due for retraining when any of these holds:

- the share of its features that drifted reaches `monitoring.drift_share`;
- its target drifted: what it predicts is no longer distributed as it learned it;
- its error on the newest period left the interval the gate accepted its champion with
  (the upper end of the champion run's `test_mae` confidence interval).

The monitor only says so, with its reasons. Retraining produces a candidate like any
other, and the gate decides whether it is served: a monitor that promoted models by
itself would be a gate that looks at no evidence.

Every run writes the per-column table to the `monitoring` layer, keeps Evidently's HTML
report beside it, and logs both to MLflow (experiment `<domain>-monitoring`).
"""

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import polars as pl
from mlflow import MlflowClient

from mlops_core.config import DomainConfig, ModelConfig
from mlops_core.provenance import code_version
from mlops_core.storage import read_table, write_table

logger = logging.getLogger(__name__)

MONITORING = "monitoring"
CHAMPION = "champion"
PREDICTION = "prediction"
REPORT_FILE = "report.html"
COLUMNS: dict[str, pl.DataType] = {
    "column": pl.String(),
    "role": pl.String(),  # feature, target or prediction
    "method": pl.String(),
    "statistic": pl.Float64(),
    "threshold": pl.Float64(),
    "drifted": pl.Boolean(),
}


@dataclass(frozen=True)
class DriftResult:
    """What the monitor found for one model, and whether it calls for retraining."""

    model: str
    reference: list[str]  # the earlier periods
    current: str  # the newest one
    columns: pl.DataFrame  # one row per column compared
    drifted_share: float  # of the model's features
    target_drifted: bool
    current_mae: float | None  # the champion's error on the newest period, if it has labels
    accepted_mae: float | None  # the upper end of the interval the gate accepted it with
    reasons: list[str]  # why it should be retrained; empty when it should not
    report_html: str

    @property
    def retrain(self) -> bool:
        return bool(self.reasons)


def periods_in_order(frame: pl.DataFrame, period: str, time: str) -> list[str]:
    """Periods ordered by when they began, never by name."""
    ordered = frame.group_by(period).agg(pl.col(time).min().alias("start")).sort("start")
    return [str(name) for name in ordered[period].to_list()]


def detect_drift(
    model: ModelConfig,
    features: pl.DataFrame,
    predictions: pl.DataFrame | None,
    drift_share: float,
    accepted_mae: float | None,
) -> DriftResult | None:
    """Compare the model's newest period with the earlier ones; None with only one period."""
    items, spec = model.items, model.spec
    periods = periods_in_order(features, items.period, items.time)
    if len(periods) < 2:
        return None
    scored = (
        features.join(predictions.select(items.id, PREDICTION), on=items.id, how="left")
        if predictions is not None
        else features
    )
    is_current = pl.col(items.period) == periods[-1]
    reference, current = scored.filter(~is_current), scored.filter(is_current)
    numeric = [*spec.numeric, spec.target] + ([PREDICTION] if predictions is not None else [])
    columns, html = column_drift(reference, current, numeric, spec.categorical)
    columns = columns.with_columns(
        pl.when(pl.col("column") == spec.target)
        .then(pl.lit("target"))
        .when(pl.col("column") == PREDICTION)
        .then(pl.lit("prediction"))
        .otherwise(pl.lit("feature"))
        .alias("role")
    ).select(list(COLUMNS))
    feature_rows = columns.filter(pl.col("role") == "feature")
    drifted_share = float(feature_rows["drifted"].mean() or 0.0)  # type: ignore[arg-type]
    target_drifted = bool(columns.filter(pl.col("role") == "target")["drifted"].any())
    current_mae = _mae(current, spec.target) if predictions is not None else None

    reasons = []
    if drifted_share >= drift_share:
        reasons.append(
            f"{drifted_share:.0%} of the features drifted (the line is {drift_share:.0%})"
        )
    if target_drifted:
        reasons.append(f"the target, {spec.target}, drifted")
    if current_mae is not None and accepted_mae is not None and current_mae > accepted_mae:
        reasons.append(
            f"the error on {periods[-1]} is {current_mae:.3f}, above the {accepted_mae:.3f} "
            "the champion was accepted with"
        )
    return DriftResult(
        model.name,
        periods[:-1],
        periods[-1],
        columns,
        drifted_share,
        target_drifted,
        current_mae,
        accepted_mae,
        reasons,
        html,
    )


def column_drift(
    reference: pl.DataFrame, current: pl.DataFrame, numeric: list[str], categorical: list[str]
) -> tuple[pl.DataFrame, str]:
    """Evidently's drift test for every column, and its HTML report.

    A column empty on either side cannot be tested and is left out, said in the log.
    """
    testable = [
        column
        for column in [*numeric, *categorical]
        if reference[column].drop_nulls().len() and current[column].drop_nulls().len()
    ]
    skipped = sorted((set(numeric) | set(categorical)) - set(testable))
    if skipped:
        logger.info("No values to test on one side, not tested: %s", skipped)
    # Evidently's UI service and collector send usage events unless this is set. The report
    # path does not import them, but nothing in this project calls home.
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(
        numerical_columns=[c for c in numeric if c in testable],
        categorical_columns=[c for c in categorical if c in testable],
    )
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(_pandas(current, testable, categorical), data_definition=definition),
        Dataset.from_pandas(_pandas(reference, testable, categorical), data_definition=definition),
    )
    rows = []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if not config["type"].endswith(":ValueDrift"):
            continue
        method, threshold, value = config["method"], float(config["threshold"]), metric["value"]
        # A p-value drifts below its threshold; a distance (Wasserstein, Jensen-Shannon)
        # drifts at or above it. This is how Evidently counts drifted columns too.
        drifted = bool(value < threshold if "p_value" in method else value >= threshold)
        rows.append(
            {
                "column": config["column"],
                "method": method,
                "statistic": float(value),
                "threshold": threshold,
                "drifted": drifted,
            }
        )
    schema = {name: dtype for name, dtype in COLUMNS.items() if name != "role"}
    return pl.DataFrame(rows, schema=schema), snapshot.get_html_str(as_iframe=False)


def accepted_error(registered_model: str) -> float | None:
    """The upper end of the test MAE interval the champion was promoted with, or None."""
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_model, CHAMPION)
        run = client.get_run(version.run_id)  # type: ignore[arg-type]
    except Exception as unavailable:  # no champion, or the registry is down
        logger.info("No champion's accepted error for %s (%s)", registered_model, unavailable)
        return None
    accepted = run.data.metrics.get("test_mae_ci_high")
    if accepted is None:  # promoted before training logged an interval
        logger.warning(
            "%s v%s was logged without a test MAE interval: its error is not checked",
            registered_model,
            version.version,
        )
    return accepted


def monitor_model(
    config: DomainConfig,
    model_name: str,
    data_dir: Path,
    tracking_uri: str,
    at: datetime | None = None,
) -> DriftResult | None:
    """Compare one model's newest period with the earlier ones, write the table and the
    report to the monitoring layer, and log both to MLflow."""
    model = config.model_named(model_name)
    features = read_table(data_dir / "features" / model.features_table)
    predictions_dir = data_dir / "predictions" / model.predictions_table
    predictions = read_table(predictions_dir) if predictions_dir.is_dir() else None
    mlflow.set_tracking_uri(tracking_uri)
    accepted = accepted_error(model.training.registered_model) if predictions is not None else None
    result = detect_drift(model, features, predictions, config.monitoring.drift_share, accepted)
    if result is None:
        logger.info("%s has one period only: nothing to compare yet", model_name)
        return None

    table = write_table(
        result.columns,
        data_dir / MONITORING / f"{model_name}_drift",
        {model.features_table: result.current},
        at or datetime.now(UTC),
    )
    report = table.parent / REPORT_FILE
    report.write_text(result.report_html, encoding="utf-8")

    mlflow.set_experiment(f"{config.name}-{MONITORING}")
    with mlflow.start_run(run_name=model_name):
        version = code_version()
        mlflow.set_tags(
            {
                "model": model_name,
                "retrain": str(result.retrain),
                "reasons": "; ".join(result.reasons),
            }
            | (version.as_tags() if version else {})
        )
        mlflow.log_params(
            {
                "reference_periods": ",".join(result.reference),
                "current_period": result.current,
                "drift_share": config.monitoring.drift_share,
            }
        )
        metrics = {
            "drifted_share": result.drifted_share,
            "target_drifted": float(result.target_drifted),
        }
        if result.current_mae is not None:
            metrics["current_mae"] = result.current_mae
        if result.accepted_mae is not None:
            metrics["accepted_mae"] = result.accepted_mae
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(table))
        mlflow.log_artifact(str(report))
    return result


def _mae(scored: pl.DataFrame, target: str) -> float | None:
    labelled = scored.drop_nulls([target, PREDICTION])
    if labelled.is_empty():
        return None
    return float((labelled[target] - labelled[PREDICTION]).abs().mean())  # type: ignore[arg-type]


def _pandas(frame: pl.DataFrame, columns: list[str], categorical: list[str]) -> pd.DataFrame:
    """The columns as Evidently reads them: numbers as floats, categories as text."""
    return frame.select(
        [
            pl.col(c).cast(pl.String) if c in categorical else pl.col(c).cast(pl.Float64)
            for c in columns
        ]
    ).to_pandas()
