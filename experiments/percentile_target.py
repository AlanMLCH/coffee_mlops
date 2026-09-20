"""Does predicting a within-period percentile rank coffees better than predicting points?

The evaluation of the champion showed that most of its test error is a level shift
between periods, not bad ranking. This experiment isolates the ranking question: train
the same pipeline on the percentile of each lot *within its own snapshot*, which has no
level, and compare how well each model orders the 2023 lots (Spearman).

Run it with the services up:  uv run python experiments/percentile_target.py
It writes one MLflow run; it is an experiment, not part of any pipeline.
"""

import logging

import mlflow
import numpy as np
from scipy.stats import spearmanr

from coffee_mlops.config import Settings, load_domain_config
from coffee_mlops.ml.evaluation import absolute_errors
from coffee_mlops.ml.targets import percentile_within, points_from_percentile
from coffee_mlops.ml.train import build_pipeline, fit_params, temporal_split, xy
from coffee_mlops.storage import read_table

PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 10}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("percentile-target")


def main() -> None:
    config = load_domain_config("coffee")
    settings = Settings()
    spec, cfg = config.model, config.training
    features = read_table(settings.data_dir / config.name / "features" / "review_features")
    train, test = temporal_split(features, cfg)

    x_train, y_train = xy(train, spec)
    x_test, y_test = xy(test, spec)
    percentile_train = percentile_within(train, spec.target, "snapshot").to_numpy()

    points_model = build_pipeline(spec, PARAMS, cfg.seed).fit(x_train, y_train, **fit_params(spec))
    rank_model = build_pipeline(spec, PARAMS, cfg.seed).fit(
        x_train, percentile_train, **fit_params(spec)
    )

    points_prediction = points_model.predict(x_test)
    rank_prediction = rank_model.predict(x_test)
    # Back to points through the training distribution: using the test one would hand the
    # model the very level shift this experiment exists to remove.
    rank_as_points = points_from_percentile(rank_prediction, y_train)

    metrics = {
        "points_model_spearman": float(spearmanr(points_prediction, y_test).statistic),
        "rank_model_spearman": float(spearmanr(rank_prediction, y_test).statistic),
        "points_model_test_mae": float(absolute_errors(y_test, points_prediction).mean()),
        "rank_model_test_mae": float(absolute_errors(y_test, rank_as_points).mean()),
        "constant_at_test_mean_mae": float(
            absolute_errors(y_test, np.full_like(y_test, y_test.mean())).mean()
        ),
    }

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(config.name)
    with mlflow.start_run(run_name="experiment-percentile-target"):
        mlflow.set_tags({"experiment": "percentile_target", "pipeline": "none"})
        mlflow.log_params(PARAMS | {"target": "percentile_within_snapshot"})
        mlflow.log_metrics(metrics)

    for name, value in metrics.items():
        logger.info("%-28s %.4f", name, value)


if __name__ == "__main__":
    main()
