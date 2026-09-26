from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor

from domains.coffee.adapter import CoffeeAdapter
from mlops_core.config import (
    AnalysisConfig,
    DomainConfig,
    GroupSplit,
    ItemsConfig,
    ModelConfig,
    ModelSpec,
    TargetBands,
    TemporalSplit,
    TrainingConfig,
)
from mlops_core.data.clean import build_clean
from mlops_core.ml.features import build_features
from mlops_core.ml.train import (
    CHAMPION,
    baseline_predictions,
    build_pipeline,
    champion_errors,
    cv_folds,
    experiment_name,
    fit_params,
    group_split,
    out_of_fold,
    promote_if_better,
    split_items,
    temporal_split,
    train_model,
    xy,
)
from mlops_core.stats import Comparison
from mlops_core.storage import write_table
from tests.fakes import with_training

REVIEW = "review"


@pytest.fixture
def fast_config(coffee_config: DomainConfig) -> DomainConfig:
    """Same config, but a tuning budget small enough for a unit test."""
    return with_training(coffee_config, REVIEW, trials=2, cv_folds=2)


@pytest.fixture
def data_dir(fast_config: DomainConfig, raw_dir: Path) -> Path:
    adapter = CoffeeAdapter(fast_config)  # type: ignore[arg-type]
    build_clean(adapter, raw_dir.parent)
    build_features(adapter, REVIEW, raw_dir.parent)
    return raw_dir.parent


def features_frame(countries: list[str | None], points: list[float]) -> pl.DataFrame:
    n = len(countries)
    return pl.DataFrame(
        {
            "grading_date": [date(2015, 1, 1 + i) for i in range(n)],
            "country": countries,
            "variety": ["caturra"] * n,
            "processing_method": ["washed"] * n,
            "color": ["green"] * n,
            **{
                c: [1.0] * n
                for c in (
                    "altitude_m",
                    "moisture_pct",
                    "category_one_defects",
                    "category_two_defects",
                    "quakers",
                    "ctx_production",
                    "ctx_arabica_share",
                    "ctx_export_share",
                    "ctx_domestic_consumption",
                )
            },
            "total_cup_points": points,
        }
    )


def test_temporal_split_never_trains_on_the_future() -> None:
    features = pl.DataFrame(
        {"grading_date": [date(2023, 1, 1), date(2010, 1, 1), date(2018, 12, 31), date(2019, 1, 1)]}
    )
    split = TemporalSplit(kind="temporal", test_from=date(2019, 1, 1), recalibration_window=30)

    train, test = temporal_split(features, split, "grading_date")

    assert train["grading_date"].to_list() == [date(2010, 1, 1), date(2018, 12, 31)]
    assert test["grading_date"].to_list() == [date(2019, 1, 1), date(2023, 1, 1)]


def test_group_baseline_falls_back_to_global_mean_for_unseen_groups(
    fast_config: DomainConfig,
) -> None:
    train = features_frame(["Mexico", "Mexico", "Kenya"], [80.0, 82.0, 84.0])
    test = features_frame(["Kenya", "Laos"], [85.0, 84.0])

    baselines = baseline_predictions(train, test, fast_config.model_named(REVIEW).spec, "country")

    assert baselines["global_mean"].tolist() == [82.0, 82.0]
    assert baselines["country_mean"].tolist() == [84.0, 82.0]
    assert "constant" not in baselines


def test_a_model_can_name_the_constant_anyone_would_forecast(fast_config: DomainConfig) -> None:
    """For a change from the month before, "no change": the random walk."""
    train = features_frame(["Mexico", "Kenya"], [80.0, 84.0])
    test = features_frame(["Kenya", "Laos"], [85.0, 84.0])
    spec = fast_config.model_named(REVIEW).spec

    baselines = baseline_predictions(train, test, spec, "country", constant=0.0)

    assert baselines["constant"].tolist() == [0.0, 0.0]


def test_pipeline_serves_unseen_and_missing_categories(fast_config: DomainConfig) -> None:
    spec = fast_config.model_named(REVIEW).spec
    train = features_frame(["Mexico", "Kenya"] * 10, [80.0, 86.0] * 10)
    pipeline = build_pipeline(spec, {"n_estimators": 10, "min_child_samples": 2}, seed=0)
    pipeline.fit(*xy(train, spec), **fit_params(spec))

    unseen = features_frame(["Narnia", None], [0.0, 0.0])
    predictions = pipeline.predict(xy(unseen, spec)[0])

    assert np.isfinite(predictions).all()


@dataclass
class FakeRegistry:
    """Just the registry calls the promotion gate makes."""

    aliases: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags[version] = value


@pytest.mark.parametrize(
    ("baseline_certainty", "champion_certainty", "promoted"),
    [
        pytest.param(0.80, None, False, id="better-but-not-sure-vs-baseline"),
        pytest.param(0.99, None, True, id="first-model-clearly-beating-baseline"),
        pytest.param(0.99, 0.60, False, id="not-clearly-better-than-champion"),
        pytest.param(0.99, 0.97, True, id="clearly-better-than-champion"),
    ],
)
def test_quality_gate_needs_evidence_not_just_a_better_average(
    baseline_certainty: float, champion_certainty: float | None, promoted: bool
) -> None:
    registry = FakeRegistry()
    comparison = Comparison(
        difference=-0.2, ci_low=-0.4, ci_high=0.1, probability_better=baseline_certainty
    )
    champion = (
        Comparison(difference=-0.1, ci_low=-0.3, ci_high=0.1, probability_better=champion_certainty)
        if champion_certainty is not None
        else None
    )

    result = promote_if_better(registry, "m", "7", comparison, champion, threshold=0.95)  # type: ignore[arg-type]

    assert result is promoted
    assert (registry.aliases.get(CHAMPION) == "7") is promoted
    assert registry.tags["7"].startswith("promoted" if promoted else "rejected")


def test_training_is_tracked_registered_and_servable(
    fast_config: DomainConfig,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)  # MLflow writes local artifacts under the working dir
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    result = train_model(fast_config, REVIEW, data_dir, tracking_uri)

    client = MlflowClient(tracking_uri)
    run = client.get_run(result.run_id)
    assert {"cv_mae", "test_mae", "test_bias", "baseline_country_mean_test_mae"} <= set(
        run.data.metrics
    )
    assert run.data.tags["features_partition"].startswith("built_at=")
    trials = client.search_runs(
        run.info.experiment_id, f"tags.mlflow.parentRunId = '{result.run_id}'"
    )
    assert len(trials) == 2

    review = fast_config.model_named(REVIEW)
    name = review.training.registered_model
    model = mlflow.sklearn.load_model(f"models:/{name}/{result.model_version}")
    features = pl.read_parquet(next(data_dir.rglob("review_features.parquet")))
    x_test = xy(split_items(features, review)[1], review.spec)[0]
    assert len(model.predict(x_test)) == 12
    assert run.data.params["split"] == "temporal"
    if result.promoted:
        assert client.get_model_version_by_alias(name, CHAMPION).version == result.model_version


def register_champion(
    tracking_uri: str, name: str, constant: float, columns: list[str] | None = None
) -> None:
    """Put a model behind the champion alias, the way a promoted training run does:
    with the input signature that says which columns it expects."""
    frame = pd.DataFrame({column: [0.0] for column in columns or ["a"]})
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("champions")
    with mlflow.start_run():
        model = DummyRegressor(strategy="constant", constant=constant).fit(frame, [constant])
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=name,
            signature=infer_signature(frame, model.predict(frame)),
            pip_requirements=["scikit-learn"],  # skip slow environment inference
        )
    MlflowClient(tracking_uri).set_registered_model_alias(
        name, CHAMPION, info.registered_model_version
    )


def test_the_champion_is_scored_on_the_very_same_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    register_champion(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", "m", constant=82.0)

    errors = champion_errors("m", pl.DataFrame({"a": [0.0, 0.0]}), np.array([80.0, 84.0]))

    assert errors is not None
    assert errors.tolist() == [2.0, 2.0]


def test_the_champion_is_fed_the_columns_it_was_trained_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature spec changes over time. The champion must still be scored on the rows,
    using its own inputs, or every candidate would look incomparable after a change."""
    monkeypatch.chdir(tmp_path)
    register_champion(
        f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", "m", 82.0, columns=["old_feature"]
    )
    # Today's table has a new feature, and keeps the old column the champion needs.
    test = pl.DataFrame({"old_feature": [0.0, 0.0], "new_feature": [1.0, 2.0]})

    errors = champion_errors("m", test, np.array([80.0, 84.0]))

    assert errors is not None
    assert errors.tolist() == [2.0, 2.0]


def test_a_champion_whose_columns_are_gone_is_not_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    register_champion(
        f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", "m", 82.0, columns=["retired_feature"]
    )

    errors = champion_errors("m", pl.DataFrame({"new_feature": [0.0]}), np.array([80.0]))

    assert errors is None
    assert "retired_feature" in caplog.text


def test_a_champion_logged_without_a_signature_is_not_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Older registries hold models logged without one; they cannot say what they need."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("champions")
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            DummyRegressor(strategy="constant", constant=82.0).fit([[0.0]], [82.0]),
            name="model",
            registered_model_name="unsigned",
            pip_requirements=["scikit-learn"],
        )
    MlflowClient(uri).set_registered_model_alias(
        "unsigned", CHAMPION, info.registered_model_version
    )

    errors = champion_errors("unsigned", pl.DataFrame({"a": [0.0]}), np.array([80.0]))

    assert errors is None
    assert "no input signature" in caplog.text


def test_without_a_champion_there_is_nothing_to_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")

    assert champion_errors("never-trained", pl.DataFrame({"a": [0.0]}), np.array([80.0])) is None


GROUPED = GroupSplit(kind="group", column="lot", test_share=0.25)


def lot_features(lots: int = 12, per_lot: int = 3) -> pl.DataFrame:
    """Items that come in families: several sizes of each lot, priced alike."""
    n = lots * per_lot
    lot = [f"lot-{i // per_lot:02d}" for i in range(n)]
    return pl.DataFrame(
        {
            "item_id": [f"item-{i:03d}" for i in range(n)],
            "period": ["2026-09"] * n,
            "observed_on": [date(2026, 9, 21)] * n,
            "lot": lot,
            "origin": [["north", "south", "east"][i // per_lot % 3] for i in range(n)],
            "size": [float(250 * (1 + i % per_lot)) for i in range(n)],
            "price": [100.0 + 10 * (i // per_lot % 3) + (i % per_lot) for i in range(n)],
        }
    )


def test_a_group_never_straddles_train_and_test() -> None:
    features = lot_features()

    train, test = group_split(features, GROUPED, "item_id", seed=7)

    assert set(train["lot"]).isdisjoint(set(test["lot"]))
    assert test["lot"].n_unique() == 3  # a quarter of the 12 lots, not of the 36 items
    assert train.height + test.height == features.height


def test_the_group_split_does_not_depend_on_row_order() -> None:
    features = lot_features()

    _, test = group_split(features, GROUPED, "item_id", seed=7)
    _, shuffled = group_split(
        features.sample(fraction=1.0, shuffle=True, seed=1), GROUPED, "item_id", seed=7
    )

    assert test.equals(shuffled)


def toy_model(split: TemporalSplit | GroupSplit = GROUPED) -> ModelConfig:
    """A model no domain has: the core must train it from config alone."""
    return ModelConfig(
        name="price",
        description="The price of a lot.",
        example={"origin": "a", "size": 1.0},
        items=ItemsConfig(table="lots", id="item_id", time="observed_on", period="period"),
        spec=ModelSpec(target="price", categorical=["origin"], numeric=["size"], leakage=[]),
        training=TrainingConfig(
            split=split,
            cv_folds=2,
            trials=2,
            seed=0,
            baseline_group="origin",
            bootstrap_resamples=200,
            min_probability_better=0.95,
            stratify_by="origin",
            min_group_size=1,
            registered_model="toy-price",
        ),
        target_bands=TargetBands(edges=[110.0], labels=["cheap", "dear"]),
    )


def test_group_folds_keep_each_group_in_one_fold() -> None:
    train = lot_features()

    folds, groups = cv_folds(train, toy_model())

    for fit_rows, score_rows in folds.split(train, groups=groups):
        assert set(groups[fit_rows]).isdisjoint(set(groups[score_rows]))


def test_a_group_split_model_trains_end_to_end_without_a_next_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No time axis to recalibrate across: the run says how it was split and skips it."""
    monkeypatch.chdir(tmp_path)
    model = toy_model()
    config = DomainConfig(
        name="toy",
        sources={},
        models=[model],
        analysis=AnalysisConfig(min_rows=1, permutation_repeats=2, published_figures=[]),
    )
    write_table(lot_features(), tmp_path / "features" / model.features_table, {})
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    result = train_model(config, "price", tmp_path, tracking_uri)

    run = MlflowClient(tracking_uri).get_run(result.run_id)
    assert run.data.params["split"] == "group" and run.data.params["split_column"] == "lot"
    assert not any(name.startswith("recalibration") for name in run.data.metrics)
    # Judged out of fold, on every group, not on one draw of a quarter of them.
    assert {"out_of_fold_mae", "out_of_fold_baseline_mae"} <= set(run.data.metrics)
    assert MlflowClient(tracking_uri).get_experiment(run.info.experiment_id).name == "toy-price"
    assert experiment_name(config, model) == "toy-price"


def test_a_group_column_that_is_also_a_feature_is_refused() -> None:
    with pytest.raises(ValueError, match="column of its own"):
        toy_model(GroupSplit(kind="group", column="origin", test_share=0.25))


def test_model_names_are_unique_and_looked_up_by_name(coffee_config: DomainConfig) -> None:
    with pytest.raises(ValueError, match="unique"):
        coffee_config.model_validate(
            coffee_config.model_dump()
            | {"models": [m.model_dump() for m in coffee_config.models] * 2}
        )
    with pytest.raises(ValueError, match="No model 'nope'"):
        coffee_config.model_named("nope")


def test_repeated_folds_draw_the_groups_again() -> None:
    """One pass over a few hundred rows is a noisy thing to choose hyperparameters by."""
    train = lot_features()
    model = toy_model()
    repeated = model.model_copy(
        update={"training": model.training.model_copy(update={"cv_repeats": 4})}
    )

    once, groups = cv_folds(train, model)
    again, _ = cv_folds(train, repeated)

    assert once.get_n_splits(groups=groups) == 2  # cv_folds
    assert again.get_n_splits(groups=groups) == 8  # cv_folds x cv_repeats
    for fit_rows, score_rows in again.split(train, groups=groups):
        assert set(groups[fit_rows]).isdisjoint(set(groups[score_rows]))


def test_every_item_is_predicted_by_a_model_that_never_saw_its_group() -> None:
    """The gate's evidence for a group model: all of it, none of it leaked."""
    features = lot_features()
    model = toy_model()

    observed, predicted, baseline = out_of_fold(features, model, {"n_estimators": 5})

    assert len(observed) == len(predicted) == len(baseline) == features.height
    assert observed.tolist() == features.sort("item_id")["price"].to_list()
    # The baseline is refitted per fold too, or it alone would have seen the held-out
    # groups: predicting each group's own mean would make it unbeatable.
    assert len(set(baseline.tolist())) > 1


def test_out_of_fold_needs_groups_to_hold_out_by() -> None:
    temporal = TemporalSplit(kind="temporal", test_from=date(2026, 1, 1), recalibration_window=5)

    with pytest.raises(TypeError, match="not split by group"):
        out_of_fold(lot_features(), toy_model(temporal), {"n_estimators": 5})


def test_the_search_space_is_narrowed_not_invented() -> None:
    training = toy_model().training

    bounds = training.model_copy(update={"search_space": {"num_leaves": (4, 8)}}).bounds
    assert bounds["num_leaves"] == (4, 8)  # the model's own
    assert bounds["n_estimators"] == (50, 800)  # the core's default, untouched

    with pytest.raises(ValidationError, match="Nothing to tune"):
        TrainingConfig.model_validate(training.model_dump() | {"search_space": {"depth": (1, 5)}})
    with pytest.raises(ValidationError, match="low to high"):
        TrainingConfig.model_validate(
            training.model_dump() | {"search_space": {"num_leaves": (16, 4)}}
        )
