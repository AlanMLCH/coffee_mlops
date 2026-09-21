"""Should the model stop looking at the features that hurt it?

The analysis pipeline measured negative permutation importance for `variety`, `country`
and `moisture_pct`: shuffling them makes the champion *better* on 2023 data. Acting on
that directly would be selecting features on the test split, so this experiment decides
on time-ordered cross-validation **inside the training period** and only then looks at
the test split once, to report what the decision cost or bought.

Run with the services up:  uv run python experiments/feature_ablation.py
"""

import logging

import mlflow
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from mlops_core.config import Settings, load_domain_config
from mlops_core.ml.evaluation import absolute_errors, compare
from mlops_core.ml.train import build_pipeline, fit_params, temporal_split, xy
from mlops_core.storage import read_table

# Fixed parameters across candidates: tuning each one separately would confound "fewer
# features" with "a luckier search", and the question here is only about the features.
PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 10}
# Measured as harmful on the test split by the analysis pipeline, worst first.
SUSPECTS = ["variety", "country", "moisture_pct"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("feature-ablation")


def candidates(
    categorical: list[str], numeric: list[str]
) -> dict[str, tuple[list[str], list[str]]]:
    """Feature sets to compare, each one dropping a bit more of what looked harmful."""
    sets = {"all features": (categorical, numeric)}
    for dropped in range(1, len(SUSPECTS) + 1):
        removed = SUSPECTS[:dropped]
        name = "without " + ", ".join(removed)
        sets[name] = (
            [column for column in categorical if column not in removed],
            [column for column in numeric if column not in removed],
        )
    return sets


def main() -> None:
    config = load_domain_config("coffee")
    settings = Settings()
    spec, cfg = config.model, config.training
    train, test = temporal_split(
        read_table(settings.data_dir / config.name / "features" / "review_features"), cfg
    )
    folds = TimeSeriesSplit(n_splits=cfg.cv_folds)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(config.name)
    scores: dict[str, float] = {}
    with mlflow.start_run(run_name="experiment-feature-ablation"):
        mlflow.set_tags({"experiment": "feature_ablation", "pipeline": "none"})
        for name, (categorical, numeric) in candidates(spec.categorical, spec.numeric).items():
            candidate = spec.model_copy(update={"categorical": categorical, "numeric": numeric})
            x_train, y_train = xy(train, candidate)
            folded = cross_val_score(
                build_pipeline(candidate, PARAMS, cfg.seed),
                x_train,
                y_train,
                cv=folds,
                scoring="neg_mean_absolute_error",
                params=fit_params(candidate),
            )
            scores[name] = float(-folded.mean())
            # MLflow metric names allow no commas, so the candidate becomes a slug.
            mlflow.log_metric(f"cv_mae_{name.replace(', ', '-').replace(' ', '_')}", scores[name])
            logger.info(
                "%-42s CV MAE %.4f  (± %.4f across folds)", name, scores[name], folded.std()
            )

        best = min(scores, key=lambda name: scores[name])
        logger.info("\nChosen by cross-validation: %s", best)

        # The test split is looked at once, after the decision, to report its price.
        full_categorical, full_numeric = candidates(spec.categorical, spec.numeric)["all features"]
        chosen_categorical, chosen_numeric = candidates(spec.categorical, spec.numeric)[best]
        errors = {}
        for name, (categorical, numeric) in (
            ("all features", (full_categorical, full_numeric)),
            (best, (chosen_categorical, chosen_numeric)),
        ):
            candidate = spec.model_copy(update={"categorical": categorical, "numeric": numeric})
            x_train, y_train = xy(train, candidate)
            x_test, y_test = xy(test, candidate)
            model = build_pipeline(candidate, PARAMS, cfg.seed).fit(
                x_train, y_train, **fit_params(candidate)
            )
            errors[name] = absolute_errors(y_test, model.predict(x_test))
            logger.info("%-42s test MAE %.4f", name, errors[name].mean())

        if best != "all features":
            verdict = compare(
                errors[best], errors["all features"], cfg.bootstrap_resamples, cfg.seed
            )
            mlflow.log_metrics(verdict.as_metrics("chosen_versus_all_features"))
            logger.info(
                "\n%s vs all features on test: %+.3f MAE "
                "(95%% CI %+.3f to %+.3f), %.0f%% sure it is better",
                best,
                verdict.difference,
                verdict.ci_low,
                verdict.ci_high,
                100 * verdict.probability_better,
            )
        mlflow.log_params({"chosen": best, "cv_folds": cfg.cv_folds} | PARAMS)


if __name__ == "__main__":
    main()
