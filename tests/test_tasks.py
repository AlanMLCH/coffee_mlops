"""What a model can predict besides a quantity: a count, a probability, a range around a
quantity - and items without a target, scored but never learned from. A toy domain no
domain has, trained end to end from config alone."""

from datetime import UTC, date, datetime
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.models import infer_signature
from pydantic import BaseModel, Field, ValidationError

from mlops_core.agent.tools import described, unstated_dates
from mlops_core.config import (
    AnalysisConfig,
    DomainConfig,
    GroupSplit,
    ItemsConfig,
    ModelConfig,
    ModelSpec,
    MonitoringConfig,
    TargetBands,
    TemporalSplit,
    TrainingConfig,
)
from mlops_core.ml.band import Band, Predicted, predicted
from mlops_core.ml.evaluation import (
    interval_scores,
    loss_name,
    poisson_deviances,
    row_losses,
    stratified_metrics,
    task_metrics,
)
from mlops_core.ml.predict import (
    HELD_OUT,
    held_out_predictions,
    predictions_schema,
    score,
)
from mlops_core.ml.registry import ServedModel
from mlops_core.ml.train import (
    CHAMPION,
    TRUSTED_MODEL_TYPES,
    baseline_bands,
    build_model,
    champion_losses,
    fit_params,
    labelled,
    out_of_fold,
    train_model,
    xy,
)
from mlops_core.serving.api import _level
from mlops_core.storage import write_table

AT = datetime(2026, 10, 2, tzinfo=UTC)
GROUPED = GroupSplit(kind="group", column="block", test_share=0.25)
YEARLY = TemporalSplit(kind="temporal", test_from=date(2024, 1, 1), recalibration_window=5)


def toy(
    spec: ModelSpec, split: GroupSplit | TemporalSplit = GROUPED, **training: object
) -> ModelConfig:
    return ModelConfig(
        name="toy",
        description="Something about a place.",
        example={"kind": "a", "people": 100.0},
        items=ItemsConfig(table="places", id="place_id", time="seen_on", period="period"),
        spec=spec,
        training=TrainingConfig.model_validate(
            {
                "split": split,
                "cv_folds": 2,
                "trials": 2,
                "seed": 0,
                "baseline_group": "kind",
                "bootstrap_resamples": 200,
                "min_probability_better": 0.95,
                "stratify_by": "kind",
                "min_group_size": 1,
                "registered_model": f"toy-{spec.task}",
            }
            | training
        ),
        target_bands=TargetBands(edges=[1.0], labels=["few", "many"]),
    )


def domain(model: ModelConfig) -> DomainConfig:
    return DomainConfig(
        name="toy",
        sources={},
        models=[model],
        analysis=AnalysisConfig(min_rows=1, permutation_repeats=2, published_figures=[]),
        monitoring=MonitoringConfig(drift_share=0.5),
    )


def places(n: int = 160, seed: int = 0) -> pl.DataFrame:
    """Places in blocks, with people and a kind; how many shops they have grows with both."""
    rng = np.random.default_rng(seed)
    people = rng.uniform(50, 500, n)
    kind = np.array(["a", "b"])[np.arange(n) % 2]
    rate = people / 100 * np.where(kind == "a", 2.0, 0.5)
    years = 2020 + np.arange(n) * 6 // n  # 2020-2025, in order
    return pl.DataFrame(
        {
            "place_id": [f"p{i:03d}" for i in range(n)],
            "period": [str(y) for y in years],
            "seen_on": [date(int(y), 6, 1) for y in years],
            "block": [f"b{i % 8}" for i in range(n)],
            "kind": kind.tolist(),
            "people": people,
            "shops": rng.poisson(rate).astype(float),
            "is_open": (rng.random(n) < np.where(kind == "a", 0.8, 0.2)).astype(float),
            "change": rng.normal(np.where(kind == "a", 5.0, -5.0), people / 50),
        }
    )


def tracking(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def trained(tmp_path: Path, model: ModelConfig, features: pl.DataFrame) -> Run:
    """Train the toy model on `features`; the run it left."""
    write_table(features, tmp_path / "features" / model.features_table, {})
    result = train_model(domain(model), model.name, tmp_path, tracking(tmp_path))
    return MlflowClient(tracking(tmp_path)).get_run(result.run_id)


COUNT = ModelSpec(
    target="shops", categorical=["kind"], numeric=["people"], leakage=[], task="count"
)
PROBABILITY = ModelSpec(
    target="is_open",
    categorical=["kind"],
    numeric=["people"],
    leakage=[],
    task="probability",
    unlabelled=True,
)
RANGE = ModelSpec(
    target="change",
    categorical=["kind"],
    numeric=["people"],
    leakage=[],
    interval=0.8,
    relative_to="people",
)


def test_a_count_is_learned_as_a_rate_and_must_beat_the_rate_per_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    model = toy(COUNT, baseline_exposure="people")

    run = trained(tmp_path, model, places())

    metrics = run.data.metrics
    assert run.data.params["task"] == "count"
    assert {
        "cv_poisson_deviance",
        "test_poisson_deviance",
        "test_mae",
        "baseline_people_rate_test_poisson_deviance",
        "out_of_fold_poisson_deviance",
        "out_of_fold_baseline_poisson_deviance",
    } <= set(metrics)


def test_a_probability_learns_only_from_the_items_whose_answer_is_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    model = toy(PROBABILITY)
    features = places().with_columns(
        pl.when(pl.col("place_id") < "p020")
        .then(None)
        .otherwise(pl.col("is_open"))
        .alias("is_open")
    )

    run = trained(tmp_path, model, features)

    params, metrics = run.data.params, run.data.metrics
    assert int(params["n_train"]) + int(params["n_test"]) == 140  # 20 have no answer
    assert {"test_brier", "test_log_loss", "test_auc", "test_base_rate", "cv_brier"} <= set(metrics)
    assert 0 <= metrics["test_mean_probability"] <= 1


def test_a_range_is_judged_by_its_interval_score_in_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    model = toy(RANGE, YEARLY, resample_by="period")

    run = trained(tmp_path, model, places())

    metrics = run.data.metrics
    assert {
        "test_coverage",
        "test_mean_width",
        "test_interval_score",
        "cv_mae",  # the point is tuned, the edges reuse what it found
        "baseline_kind_mean_test_interval_score",
        "baseline_global_mean_test_interval_score",
    } <= set(metrics)
    assert 0 <= metrics["test_coverage"] <= 1
    served = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    assert isinstance(served, Band)  # what is logged is the range, loadable as trusted


def test_a_range_and_its_level_reach_the_batch_and_the_api() -> None:
    model = toy(RANGE, YEARLY)
    features = places()
    x, y = xy(features, model.spec)
    fitted = build_model(model.spec, {"n_estimators": 10}, 0).fit(x, y, **fit_params(model.spec))
    served = ServedModel(fitted, "3", "registry")

    scored = score(features, served, model, AT)

    predictions_schema(model).validate(scored)
    assert (scored["lower"] <= scored["upper"]).all()
    level = _level("people", features.head(1), 10.0, -10.0, 20.0)
    assert level is not None
    people = features["people"][0]
    assert level.now == pytest.approx(people)
    assert level.prediction == pytest.approx(people * 1.1)
    assert level.lower == pytest.approx(people * 0.9)
    assert level.upper == pytest.approx(people * 1.2)
    assert _level(None, features.head(1), 10.0, None, None) is None
    no_value = features.head(1).with_columns(pl.lit(None, pl.Float64).alias("people"))
    assert _level("people", no_value, 10.0, None, None) is None


def test_a_grouped_model_writes_what_each_item_looks_like_to_a_model_that_never_saw_it() -> None:
    model = toy(PROBABILITY)
    features = places().with_columns(
        pl.when(pl.col("place_id") < "p004")
        .then(None)
        .otherwise(pl.col("is_open"))
        .alias("is_open")
    )
    known = labelled(features, model.spec)
    x, y = xy(known, model.spec)
    fitted = build_model(model.spec, {"n_estimators": 10}, 0).fit(x, y, **fit_params(model.spec))
    served = ServedModel(fitted, "1", "registry")

    scored = score(features, served, model, AT)

    predictions_schema(model).validate(scored)
    unknown = scored.filter(pl.col("place_id") < "p004")
    # Never learned from, so the champion's own prediction is already a held-out one.
    assert unknown[HELD_OUT].to_list() == unknown["prediction"].to_list()
    assert not scored[HELD_OUT].equals(scored["prediction"])
    with pytest.raises(TypeError, match="not split by group"):
        held_out_predictions(features, served, toy(PROBABILITY, YEARLY))


def test_a_grouped_range_is_judged_out_of_fold_with_its_edges() -> None:
    observed, scored, baseline = out_of_fold(places(), toy(RANGE), {"n_estimators": 5})

    assert scored.lower is not None and scored.upper is not None
    assert baseline.lower is not None and len(scored.upper) == len(observed)
    assert (scored.lower <= scored.upper).all()


def test_each_task_has_its_own_loss() -> None:
    y = np.array([0.0, 2.0, 5.0])

    assert loss_name(COUNT) == "poisson_deviance"
    assert loss_name(PROBABILITY) == "brier"
    assert loss_name(RANGE) == "interval_score"
    # A count of zero contributes only its rate; a rate of zero for a count is very bad.
    deviance = poisson_deviances(y, np.array([1.0, 2.0, 0.0]))
    assert deviance[0] == pytest.approx(2.0) and deviance[1] == pytest.approx(0.0)
    assert deviance[2] > 100
    # Inside, a range pays its width; outside, twice the miss over the share it may miss.
    scores = interval_scores(y, np.array([-1.0, 3.0, 0.0]), np.array([1.0, 4.0, 4.0]), 0.8)
    assert scores.tolist() == pytest.approx([2.0, 1.0 + 10.0, 4.0 + 10.0])
    with pytest.raises(ValueError, match="has none"):
        row_losses(RANGE, y, Predicted(y))
    assert row_losses(
        PROBABILITY, np.array([1.0, 0.0]), Predicted(np.array([0.5, 0.5]))
    ).tolist() == [
        0.25,
        0.25,
    ]


def test_a_probability_reports_its_ranking_only_when_there_is_something_to_rank() -> None:
    all_yes = task_metrics(PROBABILITY, np.ones(3), Predicted(np.array([0.9, 0.8, 1.0])))

    assert "auc" not in all_yes
    assert all_yes["base_rate"] == 1.0
    assert np.isfinite(all_yes["log_loss"])  # a certain, right answer is not infinite


def test_crossed_edges_still_make_a_range_and_each_group_reports_its_coverage() -> None:
    class Constant:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, x: object) -> np.ndarray:
            return np.full(2, self.value)

    band = Band(Constant(0.0), Constant(3.0), Constant(-3.0))  # upper below lower
    scored = predicted(band, None)
    assert scored.lower is not None and scored.upper is not None
    assert scored.lower.tolist() == [-3.0, -3.0] and scored.upper.tolist() == [3.0, 3.0]
    assert predicted(Constant(1.0), None).lower is None

    test = pl.DataFrame({"kind": ["a", "b"], "change": [1.0, 9.0]})
    table = stratified_metrics(test, test, scored, RANGE, "kind", 1)
    assert dict(zip(table["kind"], table["coverage"], strict=True)) == {"a": 1.0, "b": 0.0}


def test_baseline_ranges_come_from_the_training_target() -> None:
    features = places()
    (low, high), (group_low, group_high) = baseline_bands(features, features, RANGE, "kind")

    assert (low < high).all() and (group_low < group_high).all()
    with pytest.raises(ValueError, match="interval"):
        baseline_bands(features, features, COUNT, "kind")


def test_a_champion_without_a_range_is_not_compared_with_a_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(tracking(tmp_path))
    mlflow.set_experiment("champions")
    features = places()
    x, y = xy(features, RANGE)
    point = RANGE.model_copy(update={"interval": None})
    plain = build_model(point, {"n_estimators": 5}, 0).fit(x, y, **fit_params(RANGE))
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            plain,
            name="model",
            signature=infer_signature(x, y),
            registered_model_name="plain",
            skops_trusted_types=TRUSTED_MODEL_TYPES,
        )
    MlflowClient(tracking(tmp_path)).set_registered_model_alias(
        "plain", CHAMPION, str(info.registered_model_version)
    )

    assert champion_losses("plain", features, y, RANGE) is None
    assert "no range" in caplog.text


def test_what_a_task_needs_is_checked_when_the_config_is_read() -> None:
    with pytest.raises(ValidationError, match="not a count"):
        COUNT.model_validate(COUNT.model_dump() | {"interval": 0.8})
    with pytest.raises(ValidationError, match="relative_to"):
        RANGE.model_validate(RANGE.model_dump() | {"relative_to": "kind"})
    with pytest.raises(ValidationError, match="baseline_exposure"):
        toy(COUNT, baseline_exposure="kind")
    with pytest.raises(ValidationError, match="resample_by"):
        toy(COUNT, resample_by="nowhere")


class Bag(BaseModel):
    shop: str
    observed_on: date | None = Field(None, description="Defaults to today (UTC).")
    month: date


def test_a_date_the_question_never_gave_is_the_models_invention() -> None:
    request = {"shop": "a", "observed_on": "2023-10-07", "month": "2023-10-01"}

    kept = unstated_dates(Bag, request, "What would a bag at shop a cost?")

    # Optional and never stated: dropped, and the API applies its default. Required: kept.
    assert kept == {"shop": "a", "month": "2023-10-01"}
    stated = unstated_dates(Bag, request, "And in October 2023?")
    assert stated == request


def test_a_prediction_is_described_with_its_range_and_level() -> None:
    plain = described({"target": "price", "prediction": 12.345})
    ranged = described(
        {
            "target": "change_pct",
            "prediction": 5.0,
            "lower": -10.0,
            "upper": 20.0,
            "coverage": 0.8,
            "level": {
                "of": "price_last",
                "now": 300.0,
                "prediction": 315.0,
                "lower": 270.0,
                "upper": 360.0,
            },
        }
    )

    assert plain == "price = 12.35"
    assert ranged == (
        "change_pct = 5.00, between -10.00 and 20.00 80% of the time; "
        "in price_last, from 300.00 now to 315.00 (between 270.00 and 360.00)"
    )
