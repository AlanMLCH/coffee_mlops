"""Drift, per model: the newest period against every one before it.

A model's items come in periods (its config names the column: a snapshot, a catalogue
read, a decade). The newest period is compared with all the earlier ones, column by
column - every feature the model reads, its target, and its champion's predictions - with
Evidently's statistical tests, which it picks by column type and sample size (K-S or
Wasserstein for numbers, chi-squared, Z or Jensen-Shannon for categories). A column has
drifted when its test says so at Evidently's own threshold.

A retraining is decided in two steps, in this order:

1. **Are these rows new?** If a training run of the model already learned from exactly
   these feature rows (compared by content, `storage.rows_version`), nothing is due,
   whatever drifted: the same rows give the same candidate, and the gate the same answer.
   A frozen snapshot keeps its drift forever; it is retrained once, not on every run.
2. **Did they drift?** On rows no run has learned from, a retraining is due when any of
   these holds:
   - the share of its features that drifted reaches `monitoring.drift_share`;
   - its target drifted: what it predicts is no longer distributed as it learned it;
   - its error on the newest period left the interval the gate accepted its champion
     with (the upper end of the champion run's `test_mae` confidence interval).

The drift is measured and recorded either way. The monitor only says what is due:
retraining produces a candidate like any other, and the gate decides whether it is
served - a monitor that promoted models by itself would be a gate that looks at no
evidence.

Every run writes the per-column table to the `monitoring` layer, keeps Evidently's HTML
report beside it, and logs both to MLflow (experiment `<domain>-monitoring`).
"""

import logging
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import polars as pl
from mlflow import MlflowClient
from pydantic import BaseModel

from mlops_core.config import DomainConfig, ModelConfig
from mlops_core.provenance import code_version, trained_on
from mlops_core.storage import content_version, latest_partition, read_table, write_table

logger = logging.getLogger(__name__)

MONITORING = "monitoring"
CHAMPION = "champion"
PREDICTION = "prediction"
REPORT_FILE = "report.html"
VERDICT_FILE = "verdict.json"  # beside the table: what an orchestrator acts on
COLUMNS: dict[str, pl.DataType] = {
    "column": pl.String(),
    "role": pl.String(),  # feature, target or prediction
    "method": pl.String(),
    "statistic": pl.Float64(),
    "threshold": pl.Float64(),
    "drifted": pl.Boolean(),
}


class Verdict(BaseModel):
    """What the monitor decided for a model, and on which data: what a scheduler reads to
    know whether a retraining is due and has not happened yet."""

    model: str
    current: str
    retrain: bool
    reasons: list[str]  # what drifted
    data_version: str
    # The training run that already learned from these rows, if one did: then nothing is
    # due, whatever drifted. None on a verdict written before this was recorded.
    trained_run: str | None = None


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
    reasons: list[str]  # what drifted, each a reason to retrain on rows not yet learned from
    report_html: str
    data_version: str = ""  # the rows of the features compared (`storage.rows_version`)
    trained_run: str | None = None  # a training run that already learned from these rows

    @property
    def retrain(self) -> bool:
        """Due only on rows no training run has learned from, and only if they drifted."""
        return self.trained_run is None and bool(self.reasons)


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
    # An entity seen in an earlier period is one the model may have learned: only the
    # new ones say how it does on what it has not seen.
    fresh = (
        current.filter(~pl.col(items.entity).is_in(reference[items.entity].implode()))
        if items.entity
        else current
    )
    current_mae = _mae(fresh, spec.target) if predictions is not None else None
    scope = f"the {fresh.height} new items of {periods[-1]}" if items.entity else periods[-1]

    reasons = []
    if drifted_share >= drift_share:
        reasons.append(
            f"{drifted_share:.0%} of the features drifted (the line is {drift_share:.0%})"
        )
    if target_drifted:
        reasons.append(f"the target, {spec.target}, drifted")
    if current_mae is not None and accepted_mae is not None and current_mae > accepted_mae:
        reasons.append(
            f"the error on {scope} is {current_mae:.3f}, above the {accepted_mae:.3f} "
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
    features_dir = data_dir / "features" / model.features_table
    features = read_table(features_dir)
    predictions_dir = data_dir / "predictions" / model.predictions_table
    predictions = read_table(predictions_dir) if predictions_dir.is_dir() else None
    mlflow.set_tracking_uri(tracking_uri)
    # First, are these rows new? A run that already learned from them settles it.
    data_version = content_version(features_dir) or ""
    trained_run = trained_on(config, model_name, data_version) if data_version else None
    accepted = accepted_error(model.training.registered_model) if predictions is not None else None
    result = detect_drift(model, features, predictions, config.monitoring.drift_share, accepted)
    if result is None:
        logger.info("%s has one period only: nothing to compare yet", model_name)
        return None
    result = replace(result, data_version=data_version, trained_run=trained_run)

    table = write_table(
        result.columns,
        data_dir / MONITORING / f"{model_name}_drift",
        {model.features_table: result.current},
        at or datetime.now(UTC),
    )
    report = table.parent / REPORT_FILE
    report.write_text(result.report_html, encoding="utf-8")
    verdict = Verdict(
        model=model_name,
        current=result.current,
        retrain=result.retrain,
        reasons=result.reasons,
        data_version=result.data_version,
        trained_run=result.trained_run,
    )
    (table.parent / VERDICT_FILE).write_text(verdict.model_dump_json(indent=2), encoding="utf-8")

    mlflow.set_experiment(f"{config.name}-{MONITORING}")
    with mlflow.start_run(run_name=model_name):
        version = code_version()
        mlflow.set_tags(
            {
                "model": model_name,
                "retrain": str(result.retrain),
                "reasons": "; ".join(result.reasons),
                "data_version": result.data_version,
                "trained_run": result.trained_run or "",
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


def latest_verdict(data_dir: Path, model_name: str) -> Verdict | None:
    """The newest monitoring verdict for a model, or None if it was never compared."""
    partition = latest_partition(data_dir / MONITORING / f"{model_name}_drift")
    if partition is None or not (partition / VERDICT_FILE).is_file():
        return None
    return Verdict.model_validate_json((partition / VERDICT_FILE).read_text(encoding="utf-8"))


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
