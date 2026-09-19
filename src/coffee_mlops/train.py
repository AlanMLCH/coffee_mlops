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
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from coffee_mlops.config import DomainConfig, ModelSpec, TrainingConfig
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


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        # Mean over- (+) or under- (-) prediction: the level shift the model cannot see.
        "bias": float(np.mean(pred - y)),
    }


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
    client: MlflowClient, name: str, version: str, test_mae: float, gate_mae: float
) -> bool:
    """Quality gate: beat the best baseline, then beat the current champion.

    Champions are compared on test MAE; valid while the test split stays fixed
    (stage 1). Drift-aware comparison arrives with monitoring in stage 4.
    """
    if test_mae >= gate_mae:
        client.set_model_version_tag(name, version, "gate", "rejected: not better than baseline")
        logger.warning("v%s rejected: test MAE %.3f >= baseline %.3f", version, test_mae, gate_mae)
        return False
    champion_mae = float("inf")
    try:
        champion = client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        champion = None  # first model ever registered
    if champion is not None and champion.run_id is not None:
        champion_mae = client.get_run(champion.run_id).data.metrics["test_mae"]
    if test_mae >= champion_mae:
        client.set_model_version_tag(name, version, "gate", "rejected: not better than champion")
        return False
    client.set_registered_model_alias(name, CHAMPION, version)
    client.set_model_version_tag(name, version, "gate", "promoted")
    return True


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
        mlflow.set_tags({"features_partition": partition.name if partition else ""})
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

        baselines = {
            name: regression_metrics(y_test, pred)["mae"]
            for name, pred in baseline_predictions(train, test, spec, cfg.baseline_group).items()
        }
        mlflow.log_metrics({f"baseline_{name}_test_mae": mae for name, mae in baselines.items()})

        best_params, cv_mae = tune(train, spec, cfg)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        pipeline = build_pipeline(spec, best_params, cfg.seed)
        pipeline.fit(x_train, y_train, **fit_params(spec))
        predictions = pipeline.predict(x_test)
        metrics = {"cv_mae": cv_mae} | {
            f"test_{k}": v for k, v in regression_metrics(y_test, predictions).items()
        }
        mlflow.log_metrics(metrics)

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
            metrics["test_mae"],
            min(baselines.values()),
        )
    logger.info(
        "run %s -> %s v%s promoted=%s", run.info.run_id, cfg.registered_model, version, promoted
    )
    return TrainResult(
        run.info.run_id,
        version,
        promoted,
        metrics | {"best_baseline_test_mae": min(baselines.values())},
    )
