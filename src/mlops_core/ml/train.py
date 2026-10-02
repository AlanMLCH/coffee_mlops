"""Training: the model's split, baselines, Optuna-tuned LightGBM pipeline, MLflow
tracking, and a quality gate before a model version is promoted to the `champion` alias.

One named model at a time. How its items are split is part of its config: across time
when the items have a time axis worth predicting across, by group when they come in
families that must not straddle train and test.

The split also decides what the gate is allowed to read. A temporal model is judged on
the future it was asked to predict, which is the only honest test. A group model has no
such order, and holding out a quarter of the groups throws away three quarters of the
evidence: it is judged out of fold instead, on every item, each one predicted by a model
fitted without its group. That is not a softer test - no item is ever scored by a model
that saw it - it is the same test run on four times as much of the data.

What is learned depends on the model's task: a quantity, a count or a probability, all
three with LightGBM and a different objective, and a quantity can carry a range. Each is
judged by its own loss (`evaluation.row_losses`) against baselines of its own kind.

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
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from mlops_core.config import (
    DomainConfig,
    GroupSplit,
    ModelConfig,
    ModelSpec,
    Task,
    TemporalSplit,
    TrainingConfig,
)
from mlops_core.ml.band import Band, Predicted, predicted
from mlops_core.ml.evaluation import (
    absolute_errors,
    loss_name,
    mae_interval,
    recalibration_gain,
    row_losses,
    stratified_metrics,
    task_metrics,
)
from mlops_core.provenance import DATA_VERSION, code_version, experiment_name
from mlops_core.stats import Comparison, compare
from mlops_core.storage import latest_partition, read_table, rows_version

logger = logging.getLogger(__name__)

CHAMPION = "champion"
# MLflow stores sklearn models with skops, which refuses to load types it was not told
# to trust (unlike pickle, which runs arbitrary code on load). These are the LightGBM
# internals the pipeline contains, and the core's own range.
TRUSTED_MODEL_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
    "mlops_core.ml.band.Band",
]
# What LightGBM minimises for each task. All three are regressors: a cross-entropy
# regressor predicts the probability itself, so nothing downstream needs a classifier's
# `predict_proba`.
OBJECTIVES: dict[Task, str] = {
    "regression": "regression",
    "count": "poisson",
    "probability": "cross_entropy",
}
# How a trial is scored on the folds: the task's own loss, as scikit-learn names it (a
# mean squared error of a probability against 0 and 1 is its Brier score).
TUNING_SCORES: dict[Task, str] = {
    "regression": "neg_mean_absolute_error",
    "count": "neg_mean_poisson_deviance",
    "probability": "neg_mean_squared_error",
}


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_version: str
    promoted: bool
    metrics: dict[str, float]


def labelled(features: pl.DataFrame, spec: ModelSpec) -> pl.DataFrame:
    """The items whose target is known: the only ones a model learns from or is judged on."""
    return features.filter(pl.col(spec.target).is_not_null()) if spec.unlabelled else features


def split_items(features: pl.DataFrame, model: ModelConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The model's own split into train and test, as its config declares it."""
    split = model.training.split
    if isinstance(split, TemporalSplit):
        return temporal_split(features, split, model.items.time)
    return group_split(features, split, model.items.id, model.training.seed)


def temporal_split(
    features: pl.DataFrame, split: TemporalSplit, time: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Past trains, future evaluates: never a random split for data with a time axis."""
    ordered = features.sort(time)
    is_test = pl.col(time) >= split.test_from
    return ordered.filter(~is_test), ordered.filter(is_test)


def group_split(
    features: pl.DataFrame, split: GroupSplit, item_id: str, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A seeded share of the groups tests, the rest trains; no group is on both sides.

    The draw is over sorted group values, so the same data and seed give the same split
    on any machine and in any row order.
    """
    groups = sorted(features[split.column].unique().to_list())
    n_test = max(1, round(len(groups) * split.test_share))
    rng = np.random.default_rng(seed)
    tested = [groups[i] for i in rng.permutation(len(groups))[:n_test]]
    ordered = features.sort(item_id)
    is_test = pl.col(split.column).is_in(tested)
    return ordered.filter(~is_test), ordered.filter(is_test)


def out_of_fold(
    features: pl.DataFrame, model: ModelConfig, params: dict[str, Any]
) -> tuple[np.ndarray, Predicted, Predicted]:
    """Predict every item from a model fitted without its group, and the baseline the same
    way. Returns (observed, predicted, baseline prediction), aligned row by row."""
    spec, cfg = model.spec, model.training
    split = cfg.split
    if not isinstance(split, GroupSplit):  # only a group split has folds to hold out by
        raise TypeError(f"{model.name} is not split by group")
    ordered = features.sort(model.items.id)
    x, y = xy(ordered, spec)
    groups = ordered[split.column].to_numpy()
    scored, from_baseline = _Rows(len(y), spec), _Rows(len(y), spec)
    folds = GroupKFold(n_splits=cfg.cv_folds)
    for fitted_rows, held_out_rows in folds.split(x, y, groups=groups):
        fit, held_out = ordered[fitted_rows], ordered[held_out_rows]
        fitted = build_model(spec, params, cfg.seed)
        fitted.fit(x.iloc[fitted_rows], y[fitted_rows], **fit_params(spec))
        scored.put(held_out_rows, predicted(fitted, x.iloc[held_out_rows]))
        # The baseline is refitted per fold too, or it would be the only one that saw
        # the held-out groups.
        y_held = y[held_out_rows]
        from_baseline.put(
            held_out_rows,
            min(
                baselines(fit, held_out, spec, cfg).values(),
                key=lambda b: float(row_losses(spec, y_held, b).mean()),
            ),
        )
    return y, scored.done(), from_baseline.done()


class _Rows:
    """Predictions assembled fold by fold into one set, aligned with the rows."""

    def __init__(self, n: int, spec: ModelSpec) -> None:
        self.point = np.zeros(n)
        self.lower = np.zeros(n) if spec.interval is not None else None
        self.upper = np.zeros(n) if spec.interval is not None else None

    def put(self, rows: np.ndarray, scored: Predicted) -> None:
        self.point[rows] = scored.point
        if self.lower is not None and self.upper is not None:
            self.lower[rows], self.upper[rows] = scored.lower, scored.upper

    def done(self) -> Predicted:
        return Predicted(self.point, self.lower, self.upper)


def cv_folds(train: pl.DataFrame, model: ModelConfig) -> tuple[BaseCrossValidator, Any]:
    """Folds inside the training split, of the same kind as the split itself: a model
    tuned on random folds would be tuned for a problem it is not evaluated on.

    With `cv_repeats` the groups are drawn again, `repeats` times over: on a few hundred
    rows one pass is noisy enough that the tuner ranks draws instead of models.
    """
    cfg = model.training
    folds, repeats = cfg.cv_folds, cfg.cv_repeats
    if isinstance(cfg.split, TemporalSplit):
        return TimeSeriesSplit(n_splits=folds), None  # `train` is already in time order
    groups = train[cfg.split.column].to_numpy()
    if repeats == 1:
        return GroupKFold(n_splits=folds), groups
    return (
        GroupShuffleSplit(n_splits=folds * repeats, test_size=1 / folds, random_state=cfg.seed),
        groups,
    )


def build_model(spec: ModelSpec, params: dict[str, Any], seed: int) -> Pipeline | Band:
    """What the model's spec serves: one pipeline, or with an `interval` a point and the
    two quantile pipelines at the edges of its range, all with the same hyperparameters
    (tuned for the point: tuning each edge would triple the search for a second-order gain)."""
    point = build_pipeline(spec, params, seed)
    if spec.interval is None:
        return point
    tail = (1 - spec.interval) / 2
    return Band(
        point,
        build_pipeline(spec, params, seed, quantile=tail),
        build_pipeline(spec, params, seed, quantile=1 - tail),
    )


def build_pipeline(
    spec: ModelSpec, params: dict[str, Any], seed: int, quantile: float | None = None
) -> Pipeline:
    """Encoder and model are fitted together, so rare-category grouping is learned on
    the training split only and travels with the model to serving. With a `quantile` the
    model learns that quantile of the target instead of the task's own objective."""
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
    objective: dict[str, Any] = (
        {"objective": "quantile", "alpha": quantile}
        if quantile is not None
        else {"objective": OBJECTIVES[spec.task]}
    )
    model = LGBMRegressor(random_state=seed, verbose=-1, **objective, **model_params)
    return Pipeline([("prep", prep), ("model", model)])


def fit_params(spec: ModelSpec) -> dict[str, Any]:
    # The ColumnTransformer puts the encoded categoricals first.
    return {"model__categorical_feature": list(range(len(spec.categorical)))}


def xy(df: pl.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, np.ndarray]:
    return df.select(spec.features).to_pandas(), df[spec.target].to_numpy()


def baseline_predictions(
    train: pl.DataFrame,
    test: pl.DataFrame,
    spec: ModelSpec,
    group: str,
    constant: float | None = None,
    exposure: str | None = None,
) -> dict[str, np.ndarray]:
    """What anyone could predict without a model: the training mean, the mean of the
    item's group, and - when the model names them - a constant, such as "no change", and
    the training rate per unit of an exposure times the item's own. For a probability
    the means are rates of yes, so every baseline is a probability too."""
    overall = float(train[spec.target].mean())  # type: ignore[arg-type]
    group_means = train.group_by(group).agg(pl.col(spec.target).mean().alias("_pred"))
    by_group = test.join(group_means, on=group, how="left")["_pred"].fill_null(overall)
    predictions = {
        "global_mean": np.full(test.height, overall),
        f"{group}_mean": by_group.to_numpy(),
    }
    if constant is not None:
        predictions["constant"] = np.full(test.height, constant)
    if exposure is not None:
        known = train.drop_nulls(exposure)
        rate = float(known[spec.target].sum()) / float(known[exposure].sum())
        per_unit = (test[exposure] * rate).fill_null(overall)
        predictions[f"{exposure}_rate"] = per_unit.to_numpy()
    return predictions


def baseline_bands(
    train: pl.DataFrame, test: pl.DataFrame, spec: ModelSpec, group: str
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The range anyone could give without a model: the training target's own quantiles,
    over all items and within the item's group. For a change it is the historical range,
    which is what a forecast's range has to be narrower than while still holding."""
    if spec.interval is None:
        raise ValueError("Only a model with an interval has a range to compare")
    tail = (1 - spec.interval) / 2
    target = pl.col(spec.target)
    low, high = (
        float(train[spec.target].quantile(tail)),  # type: ignore[arg-type]
        float(train[spec.target].quantile(1 - tail)),  # type: ignore[arg-type]
    )
    edges = train.group_by(group).agg(
        target.quantile(tail).alias("_low"), target.quantile(1 - tail).alias("_high")
    )
    by_group = test.join(edges, on=group, how="left")
    return (
        (np.full(test.height, low), np.full(test.height, high)),
        (
            by_group["_low"].fill_null(low).to_numpy(),
            by_group["_high"].fill_null(high).to_numpy(),
        ),
    )


def baselines(
    train: pl.DataFrame, test: pl.DataFrame, spec: ModelSpec, cfg: TrainingConfig
) -> dict[str, Predicted]:
    """Every baseline as a prediction of the model's own kind: with an interval, the
    group's mean comes with the group's range and every other one with the overall range."""
    points = baseline_predictions(
        train, test, spec, cfg.baseline_group, cfg.baseline_constant, cfg.baseline_exposure
    )
    if spec.interval is None:
        return {name: Predicted(point) for name, point in points.items()}
    overall, by_group = baseline_bands(train, test, spec, cfg.baseline_group)
    grouped = f"{cfg.baseline_group}_mean"
    return {
        name: Predicted(point, *(by_group if name == grouped else overall))
        for name, point in points.items()
    }


def tune(train: pl.DataFrame, model: ModelConfig) -> tuple[dict[str, Any], float]:
    """TPE search over the model's CV folds; each trial is a nested MLflow run. A model
    with a range is tuned for its point: the edges reuse what the point found."""
    spec, cfg = model.spec, model.training
    x, y = xy(train, spec)
    folds, groups = cv_folds(train, model)

    bounds = cfg.bounds

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", *bounds["learning_rate"], log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", *_whole(bounds["n_estimators"])),
            "num_leaves": trial.suggest_int("num_leaves", *_whole(bounds["num_leaves"])),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *_whole(bounds["min_child_samples"])
            ),
            "reg_lambda": trial.suggest_float("reg_lambda", *bounds["reg_lambda"], log=True),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *bounds["colsample_bytree"]
            ),
            "min_frequency": trial.suggest_int("min_frequency", *_whole(bounds["min_frequency"])),
        }
        scores = cross_val_score(
            build_pipeline(spec, params, cfg.seed),
            x,
            y,
            groups=groups,
            cv=folds,
            scoring=TUNING_SCORES[spec.task],
            params=fit_params(spec),
        )
        cv_loss = float(-scores.mean())
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric(cv_metric(spec), cv_loss)
        return cv_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.trials)
    return study.best_params, study.best_value


def cv_metric(spec: ModelSpec) -> str:
    """The name of the loss the folds were scored by: the point's, for a range."""
    return f"cv_{loss_name(spec.model_copy(update={'interval': None}))}"


def _whole(bounds: tuple[float, float]) -> tuple[int, int]:
    """A count's bounds, as the counts they name."""
    low, high = bounds
    return int(low), int(high)


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


def champion_losses(
    name: str, test: pl.DataFrame, y_test: np.ndarray, spec: ModelSpec
) -> np.ndarray | None:
    """The current champion's losses on the same rows, or None if it cannot be scored on
    them - or not by the same loss: a champion without a range cannot be compared with a
    candidate judged by its range.

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
    scored = predicted(champion, test.select(columns).to_pandas())
    if spec.interval is not None and scored.lower is None:
        logger.warning("The champion predicts no range: promoting on the baseline alone")
        return None
    return row_losses(spec, y_test, scored)


def train_model(
    config: DomainConfig, model_name: str, data_dir: Path, tracking_uri: str
) -> TrainResult:
    model = config.model_named(model_name)
    spec, cfg, split = model.spec, model.training, model.training.split
    loss = loss_name(spec)
    table_dir = data_dir / "features" / model.features_table
    every_item = read_table(table_dir)
    features = labelled(every_item, spec)
    partition = latest_partition(table_dir)
    train, test = split_items(features, model)
    x_train, y_train = xy(train, spec)
    x_test, y_test = xy(test, spec)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name(config, model))
    with mlflow.start_run(run_name=f"{spec.target}-lightgbm") as run:
        # Provenance: MLflow records the entry point but not the revision, and a run
        # made from a dirty tree cannot be reproduced.
        version_tags = code_version()
        mlflow.set_tags(
            {
                "features_partition": partition.name if partition else "",
                # The rows it learned from: retraining on the same rows is pointless.
                # Unlabelled items count too: scoring new ones is what the model is for.
                DATA_VERSION: rows_version(every_item),
            }
            | (version_tags.as_tags() if version_tags else {})
        )
        mlflow.log_params(
            {
                "target": spec.target,
                "task": spec.task,
                "interval": spec.interval or "",
                "categorical": ",".join(spec.categorical),
                "numeric": ",".join(spec.numeric),
                "n_train": train.height,
                "n_test": test.height,
            }
            | split_params(split)
        )
        source = str(table_dir)
        mlflow.log_input(from_pandas(x_train, source=source, name="train"), "training")
        mlflow.log_input(from_pandas(x_test, source=source, name="test"), "testing")

        scored_baselines = baselines(train, test, spec, cfg)
        baseline_losses = {
            name: row_losses(spec, y_test, scored) for name, scored in scored_baselines.items()
        }
        mlflow.log_metrics(
            {
                f"baseline_{name}_test_mae": float(absolute_errors(y_test, scored.point).mean())
                for name, scored in scored_baselines.items()
            }
            | {
                f"baseline_{name}_test_{loss}": float(e.mean())
                for name, e in baseline_losses.items()
            }
        )
        # The gate compares against the strongest baseline, not the most flattering one.
        best_baseline = min(baseline_losses.values(), key=lambda e: e.mean())

        best_params, cv_loss = tune(train, model)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        fitted = build_model(spec, best_params, cfg.seed)
        fitted.fit(x_train, y_train, **fit_params(spec))
        scored = predicted(fitted, x_test)
        predictions = scored.point
        losses = row_losses(spec, y_test, scored)
        families = resampled_together(test, model)
        resamples, seed = cfg.bootstrap_resamples, cfg.seed
        ci_low, ci_high = mae_interval(
            absolute_errors(y_test, predictions), resamples, seed, families
        )
        versus_baseline = compare(losses, best_baseline, resamples, seed, families)
        champion = champion_losses(cfg.registered_model, test, y_test, spec)
        versus_champion = (
            compare(losses, champion, resamples, seed, families) if champion is not None else None
        )
        # A group model's evidence against the baseline is gathered on every group.
        out_of_fold_metrics: dict[str, float] = {}
        if isinstance(split, GroupSplit):
            observed, predicted_oof, baseline_oof = out_of_fold(features, model, best_params)
            oof_losses = row_losses(spec, observed, predicted_oof)
            oof_baseline = row_losses(spec, observed, baseline_oof)
            all_families = resampled_together(features.sort(model.items.id), model)
            versus_baseline = compare(oof_losses, oof_baseline, resamples, seed, all_families)
            out_of_fold_metrics = {
                f"out_of_fold_{loss}": float(oof_losses.mean()),
                f"out_of_fold_baseline_{loss}": float(oof_baseline.mean()),
            }

        metrics = (
            {
                cv_metric(spec): cv_loss,
                "test_mae_ci_low": ci_low,
                "test_mae_ci_high": ci_high,
            }
            | {f"test_{k}": v for k, v in task_metrics(spec, y_test, scored).items()}
            | versus_baseline.as_metrics("versus_baseline")
            | out_of_fold_metrics
            | (versus_champion.as_metrics("versus_champion") if versus_champion else {})
            | (
                recalibration_gain(
                    test, predictions, spec, split.recalibration_window, model.items.time
                )
                if isinstance(split, TemporalSplit)
                else {}  # no next period to recalibrate on
            )
        )
        mlflow.log_metrics(metrics)
        # Logged as a table, not as metrics: one row per group, and group names change.
        mlflow.log_table(
            stratified_metrics(
                train, test, scored, spec, cfg.stratify_by, cfg.min_group_size
            ).to_pandas(),
            artifact_file="stratified_metrics.json",
        )

        info = mlflow.sklearn.log_model(
            fitted,
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
        metrics | {f"best_baseline_test_{loss}": float(best_baseline.mean())},
    )


def resampled_together(rows: pl.DataFrame, model: ModelConfig) -> np.ndarray | None:
    """What the gate's bootstrap resamples whole, row by row: the `resample_by` blocks
    when the model names them, else a group split's families - one product priced wrong
    is one mistake, not five - else nothing, and rows are resampled one by one."""
    cfg = model.training
    if cfg.resample_by is not None:
        return rows[cfg.resample_by].to_numpy()
    if isinstance(cfg.split, GroupSplit):
        return rows[cfg.split.column].to_numpy()
    return None


def split_params(split: TemporalSplit | GroupSplit) -> dict[str, str | float]:
    """How the run was split, as MLflow params: two runs split differently do not compare."""
    if isinstance(split, TemporalSplit):
        return {"split": split.kind, "test_from": split.test_from.isoformat()}
    return {"split": split.kind, "split_column": split.column, "test_share": split.test_share}
