"""Training: temporal split, baselines, Optuna-tuned LightGBM pipeline, MLflow tracking,
and a quality gate before a model version is promoted to the `champion` alias.

sklearn, LightGBM and MLflow speak pandas, so frames cross to pandas at this boundary.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from coffee_mlops.config import DomainConfig, ModelSpec, TrainingConfig
from coffee_mlops.ml.evaluation import (
    Comparison,
    absolute_errors,
    compare,
    mae_interval,
    recalibration_gain,
    regression_metrics,
    stratified_metrics,
)
from coffee_mlops.provenance import code_version
from coffee_mlops.storage import latest_partition, read_table

logger = logging.getLogger(__name__)

CHAMPION = "champion"
# MLflow stores sklearn models with skops, which refuses to load types it was not told
# to trust (unlike pickle, which runs arbitrary code on load). These are the LightGBM
# internals the pipeline contains.
TRUSTED_MODEL_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
]


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_version: str
    promoted: bool
    metrics: dict[str, float]


def temporal_split(
    features: pl.DataFrame, cfg: TrainingConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Past trains, future evaluates: never a random split for data with a time axis."""
    ordered = features.sort("grading_date")
    is_test = pl.col("grading_date") >= cfg.test_from
    return ordered.filter(~is_test), ordered.filter(is_test)


def build_pipeline(spec: ModelSpec, params: dict[str, Any], seed: int) -> Pipeline:
    """Encoder and model are fitted together, so rare-category grouping is learned on
    the training split only and travels with the model to serving."""
    model_params = dict(params)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,  # unseen at training time -> treated as missing
        encoded_missing_value=np.nan,
        min_frequency=model_params.pop("min_frequency", 5),
    )
    prep = ColumnTransformer(
        [("categorical", encoder, spec.categorical)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = LGBMRegressor(random_state=seed, verbose=-1, **model_params)
    return Pipeline([("prep", prep), ("model", model)])


def fit_params(spec: ModelSpec) -> dict[str, Any]:
    # The ColumnTransformer puts the encoded categoricals first.
    return {"model__categorical_feature": list(range(len(spec.categorical)))}


def xy(df: pl.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, np.ndarray]:
    return df.select(spec.features).to_pandas(), df[spec.target].to_numpy()


def baseline_predictions(
    train: pl.DataFrame, test: pl.DataFrame, spec: ModelSpec, group: str
) -> dict[str, np.ndarray]:
    overall = float(train[spec.target].mean())  # type: ignore[arg-type]
    group_means = train.group_by(group).agg(pl.col(spec.target).mean().alias("_pred"))
    by_group = test.join(group_means, on=group, how="left")["_pred"].fill_null(overall)
    return {
        "global_mean": np.full(test.height, overall),
        f"{group}_mean": by_group.to_numpy(),
    }


def tune(train: pl.DataFrame, spec: ModelSpec, cfg: TrainingConfig) -> tuple[dict[str, Any], float]:
    """TPE search over time-ordered CV folds; each trial is a nested MLflow run."""
    x, y = xy(train, spec)
    folds = TimeSeriesSplit(n_splits=cfg.cv_folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_frequency": trial.suggest_int("min_frequency", 2, 30),
        }
        scores = cross_val_score(
            build_pipeline(spec, params, cfg.seed),
            x,
            y,
            cv=folds,
            scoring="neg_mean_absolute_error",
            params=fit_params(spec),
        )
        cv_mae = float(-scores.mean())
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_mae", cv_mae)
        return cv_mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.trials)
    return study.best_params, study.best_value


def promote_if_better(
    client: MlflowClient,
    name: str,
    version: str,
    baseline: Comparison,
    champion: Comparison | None,
    threshold: float,
) -> bool:
    """Quality gate: beat the best baseline, then the current champion, with evidence.

    Both comparisons are paired bootstraps on the same test rows, so a candidate is
    promoted only when it wins in at least `threshold` of the resamples. A better point
    estimate is not enough: on a small test split that is often noise.
    """
    for reason, comparison in (("baseline", baseline), ("champion", champion)):
        if comparison is None:
            continue  # nothing registered to compare against yet
        if comparison.probability_better < threshold:
            note = f"rejected: only {comparison.probability_better:.0%} sure it beats the {reason}"
            client.set_model_version_tag(name, version, "gate", note)
            logger.warning("v%s %s", version, note)
            return False
    client.set_registered_model_alias(name, CHAMPION, version)
    client.set_model_version_tag(name, version, "gate", "promoted")
    return True


def champion_errors(name: str, test: pl.DataFrame, y_test: np.ndarray) -> np.ndarray | None:
    """Absolute errors of the current champion on the same rows, or None if it cannot
    be scored on them.

    The champion is fed **its own** input columns, read from the signature it was logged
    with, not today's feature list. Without that, changing the feature spec would make
    every new model incomparable to the one in production, which is precisely when a
    comparison matters most. Columns a champion needs survive in the feature table as
    metadata; if one is truly gone, the comparison is skipped and said out loud.
    """
    uri = f"models:/{name}@{CHAMPION}"
    try:
        champion = mlflow.sklearn.load_model(uri)
        signature = mlflow.models.get_model_info(uri).signature
    except Exception as unavailable:  # no alias yet, or registry unreachable
        logger.info("No champion to compare against (%s)", unavailable)
        return None
    if signature is None:  # a model logged without one cannot say what it needs
        logger.warning("The champion has no input signature: skipping the comparison")
        return None
    columns = [column.name for column in signature.inputs.inputs]
    missing = [column for column in columns if column not in test.columns]
    if missing:
        logger.warning(
            "The champion needs %s, which this feature table no longer has: "
            "promoting on the baseline comparison alone",
            missing,
        )
        return None
    return absolute_errors(y_test, champion.predict(test.select(columns).to_pandas()))


def train_model(config: DomainConfig, data_dir: Path, tracking_uri: str) -> TrainResult:
    spec, cfg = config.model, config.training
    table_dir = data_dir / "features" / "review_features"
    features = read_table(table_dir)
    partition = latest_partition(table_dir)
    train, test = temporal_split(features, cfg)
    x_train, y_train = xy(train, spec)
    x_test, y_test = xy(test, spec)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.name)
    with mlflow.start_run(run_name=f"{spec.target}-lightgbm") as run:
        # Provenance: MLflow records the entry point but not the revision, and a run
        # made from a dirty tree cannot be reproduced.
        version_tags = code_version()
        mlflow.set_tags(
            {"features_partition": partition.name if partition else ""}
            | (version_tags.as_tags() if version_tags else {})
        )
        mlflow.log_params(
            {
                "target": spec.target,
                "categorical": ",".join(spec.categorical),
                "numeric": ",".join(spec.numeric),
                "test_from": cfg.test_from.isoformat(),
                "n_train": train.height,
                "n_test": test.height,
            }
        )
        source = str(table_dir)
        mlflow.log_input(from_pandas(x_train, source=source, name="train"), "training")
        mlflow.log_input(from_pandas(x_test, source=source, name="test"), "testing")

        baseline_errors = {
            name: absolute_errors(y_test, pred)
            for name, pred in baseline_predictions(train, test, spec, cfg.baseline_group).items()
        }
        mlflow.log_metrics(
            {f"baseline_{name}_test_mae": float(e.mean()) for name, e in baseline_errors.items()}
        )
        # The gate compares against the strongest baseline, not the most flattering one.
        best_baseline = min(baseline_errors.values(), key=lambda e: e.mean())

        best_params, cv_mae = tune(train, spec, cfg)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        pipeline = build_pipeline(spec, best_params, cfg.seed)
        pipeline.fit(x_train, y_train, **fit_params(spec))
        predictions = pipeline.predict(x_test)
        errors = absolute_errors(y_test, predictions)
        ci_low, ci_high = mae_interval(errors, cfg.bootstrap_resamples, cfg.seed)
        versus_baseline = compare(errors, best_baseline, cfg.bootstrap_resamples, cfg.seed)
        champion = champion_errors(cfg.registered_model, test, y_test)
        versus_champion = (
            compare(errors, champion, cfg.bootstrap_resamples, cfg.seed)
            if champion is not None
            else None
        )

        metrics = (
            {"cv_mae": cv_mae, "test_mae_ci_low": ci_low, "test_mae_ci_high": ci_high}
            | {f"test_{k}": v for k, v in regression_metrics(y_test, predictions).items()}
            | versus_baseline.as_metrics("versus_baseline")
            | (versus_champion.as_metrics("versus_champion") if versus_champion else {})
            | recalibration_gain(test, predictions, spec, cfg.recalibration_window)
        )
        mlflow.log_metrics(metrics)
        # Logged as a table, not as metrics: one row per group, and group names change.
        mlflow.log_table(
            stratified_metrics(
                train, test, predictions, spec, cfg.stratify_by, cfg.min_group_size
            ).to_pandas(),
            artifact_file="stratified_metrics.json",
        )

        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(x_test, predictions),
            input_example=x_test.head(3),
            registered_model_name=cfg.registered_model,
            skops_trusted_types=TRUSTED_MODEL_TYPES,
        )
        version = str(info.registered_model_version)
        promoted = promote_if_better(
            MlflowClient(),
            cfg.registered_model,
            version,
            versus_baseline,
            versus_champion,
            cfg.min_probability_better,
        )
    logger.info(
        "run %s -> %s v%s promoted=%s", run.info.run_id, cfg.registered_model, version, promoted
    )
    return TrainResult(
        run.info.run_id,
        version,
        promoted,
        metrics | {"best_baseline_test_mae": float(best_baseline.mean())},
    )
