"""Can anything price a kilo better than "what this shop usually charges"?

The gate refused the first price model: 319 MXN/kg against the shop-mean baseline's 326,
60% sure where 95% is asked. Before changing the core to support another estimator, this
measures whether another estimator would help at all, on the model's own split and with
the gate's own paired bootstrap - resampling coffees, not bags.

Four candidates: the tuned LightGBM the pipeline builds, the same on a log target (prices
are right-skewed and a few Geshas pull the mean), a ridge on one-hot columns (few rows,
mostly categorical) and a small random forest. Run with:

    uv run python experiments/price_estimators.py
"""

import logging

import mlflow
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_core.adapter import load_adapter
from mlops_core.config import GroupSplit, Settings
from mlops_core.ml.evaluation import absolute_errors
from mlops_core.ml.train import (
    baseline_predictions,
    build_pipeline,
    experiment_name,
    fit_params,
    split_items,
    tune,
    xy,
)
from mlops_core.stats import compare
from mlops_core.storage import read_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("price-estimators")

MODEL = "offer"


def linear(spec: object) -> Pipeline:
    """Ridge on one-hot columns: with 118 training coffees, mostly categorical, a
    regularised linear model is the honest thing to compare a boosted tree against."""
    categorical, numeric = spec.categorical, spec.numeric  # type: ignore[attr-defined]
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=3,
                                sparse_output=False,
                            ),
                            categorical,
                        ),
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                    ]
                ),
            ),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 30))),
        ]
    )


def forest(spec: object, seed: int) -> Pipeline:
    categorical, numeric = spec.categorical, spec.numeric  # type: ignore[attr-defined]
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=3,
                                sparse_output=False,
                            ),
                            categorical,
                        ),
                        ("numeric", SimpleImputer(strategy="median"), numeric),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=seed
                ),
            ),
        ]
    )


def main() -> None:
    config = load_adapter("coffee").config
    model = config.model_named(MODEL)
    spec, cfg = model.spec, model.training
    settings = Settings()
    features = read_table(settings.data_dir / config.name / "features" / model.features_table)
    train, test = split_items(features, model)
    x_train, y_train = xy(train, spec)
    x_test, y_test = xy(test, spec)
    grouped = cfg.split
    assert isinstance(grouped, GroupSplit)
    families = test[grouped.column].to_numpy()

    baselines = baseline_predictions(train, test, spec, cfg.baseline_group)
    best_baseline = min(
        (absolute_errors(y_test, prediction) for prediction in baselines.values()),
        key=lambda errors: errors.mean(),
    )
    logger.info(
        "baseline MAE %.1f on %d offers of %d coffees",
        best_baseline.mean(),
        test.height,
        test[grouped.column].n_unique(),
    )

    best_params, _ = tune(train, model)
    candidates = {
        "lightgbm (tuned)": (build_pipeline(spec, best_params, cfg.seed), False),
        "lightgbm on log price": (build_pipeline(spec, best_params, cfg.seed), True),
        "ridge on one-hot": (linear(spec), False),
        "ridge on log price": (linear(spec), True),
        "random forest": (forest(spec, cfg.seed), False),
    }

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name(config, model))
    rows = []
    for name, (pipeline, log_target) in candidates.items():
        fitted = pipeline.fit(
            x_train,
            np.log(y_train) if log_target else y_train,
            **(fit_params(spec) if "lightgbm" in name else {}),
        )
        predicted = fitted.predict(x_test)
        if log_target:
            predicted = np.exp(predicted)
        errors = absolute_errors(y_test, predicted)
        versus = compare(errors, best_baseline, cfg.bootstrap_resamples, cfg.seed, families)
        rows.append(
            {
                "candidate": name,
                "test_mae": float(errors.mean()),
                "versus_baseline": versus.difference,
                "probability_better": versus.probability_better,
            }
        )
        logger.info(
            "%-24s MAE %6.1f  vs baseline %+7.1f  %3.0f%% sure",
            name,
            errors.mean(),
            versus.difference,
            100 * versus.probability_better,
        )
    with mlflow.start_run(run_name="price-estimators"):
        mlflow.log_table(pl.DataFrame(rows).to_pandas(), artifact_file="price_estimators.json")


if __name__ == "__main__":
    main()
