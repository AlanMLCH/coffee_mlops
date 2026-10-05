"""A model with no champion is served the best there is: a mediocre model is better than
none, but never one worse than the rule anyone would use. The candidate when on average it
beats every baseline the API could serve; else the best of those baselines, registered as a
version of its own. Either way the version, the API and the agent say it is provisional."""

from datetime import date, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from mlops_core.agent.tools import described
from mlops_core.config import (
    AnalysisConfig,
    DomainConfig,
    ItemsConfig,
    ModelConfig,
    ModelSpec,
    MonitoringConfig,
    TargetBands,
    TemporalSplit,
    TrainingConfig,
)
from mlops_core.ml.band import Band, predicted
from mlops_core.ml.baseline_models import Lookup, Rate, lookup_key
from mlops_core.ml.registry import load_champion
from mlops_core.ml.train import (
    CHAMPION,
    PROVISIONAL,
    baselines,
    has_champion,
    promote_provisional,
    servable_baselines,
    train_model,
)
from mlops_core.stats import Comparison
from mlops_core.storage import write_table


def spec(interval: float | None = None) -> ModelSpec:
    return ModelSpec(
        target="y", categorical=["slot"], numeric=["size"], leakage=[], interval=interval
    )


def training(**overrides: object) -> TrainingConfig:
    base: dict[str, object] = {
        "split": TemporalSplit(kind="temporal", test_from=date(2025, 3, 1), recalibration_window=5),
        "cv_folds": 2,
        "trials": 2,
        "seed": 0,
        "baseline_group": "slot",
        "bootstrap_resamples": 200,
        "min_probability_better": 0.95,
        "stratify_by": "slot",
        "min_group_size": 1,
        "registered_model": "toy-y",
    }
    return TrainingConfig.model_validate(base | overrides)


def rows(n: int, seed: int = 0) -> pl.DataFrame:
    """Two slots, a size, and a target that is the slot's level plus noise."""
    rng = np.random.default_rng(seed)
    slot = ["am" if i % 2 else "pm" for i in range(n)]
    return pl.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(n)],
            "day": [date(2025, 1, 1) + timedelta(days=i) for i in range(n)],
            "period": [f"2025-{1 + i // 31:02d}" for i in range(n)],
            "slot": slot,
            "size": [float(1 + i % 3) for i in range(n)],
            "y": [(10.0 if s == "am" else 4.0) + float(rng.normal(0, 1)) for s in slot],
        }
    )


def test_a_served_baseline_predicts_what_the_gate_measured() -> None:
    train, test = rows(60), rows(20, seed=1)
    cfg = training(baseline_constant=0.0, baseline_exposure="size")
    model_spec = ModelSpec(target="y", categorical=["slot"], numeric=["size"], leakage=[])

    servable = servable_baselines(train, model_spec, cfg)
    measured = baselines(train, test, model_spec, cfg)

    assert set(servable) == {"global_mean", "slot_mean", "constant", "size_rate"}
    x = test.select(model_spec.features).to_pandas()
    for name, model in servable.items():
        assert np.allclose(model.predict(x), measured[name].point), name


def test_a_served_range_baseline_has_the_gates_edges() -> None:
    train, test = rows(60), rows(20, seed=1)
    cfg, ranged = training(), spec(interval=0.8)

    servable = servable_baselines(train, ranged, cfg)
    measured = baselines(train, test, ranged, cfg)

    x = test.select(ranged.features).to_pandas()
    for name, model in servable.items():
        assert isinstance(model, Band)
        served, gate = predicted(model, x), measured[name]
        assert np.allclose(served.point, gate.point)
        assert np.allclose(served.lower, gate.lower) and np.allclose(served.upper, gate.upper)  # type: ignore[arg-type]


def test_a_baseline_of_a_column_the_model_does_not_read_is_not_served() -> None:
    cfg = training(baseline_group="period")  # a key, not an input

    assert set(servable_baselines(rows(30), spec(), cfg)) == {"global_mean"}


def test_a_lookup_reads_a_group_however_the_api_typed_it() -> None:
    lookup = Lookup("months", {lookup_key(3): 1.5, lookup_key("other"): 9.0}, default=0.0)
    rate = Rate("people", rate=0.5, default=7.0)

    assert lookup.predict(pd.DataFrame({"months": [3, 3.0, 6.0]})).tolist() == [1.5, 1.5, 0.0]
    assert Lookup(None, {}, 4.0).predict(pd.DataFrame({"a": [1, 2]})).tolist() == [4.0, 4.0]
    assert rate.predict(pd.DataFrame({"people": [10.0, None]})).tolist() == [5.0, 7.0]
    assert lookup.fit(None, None) is lookup and rate.fit(None, None) is rate


class Registry:
    """What promotion writes: aliases and version tags."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.tags: dict[str, dict[str, str]] = {}

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags.setdefault(version, {})[key] = value


UNSURE = Comparison(difference=-0.1, ci_low=-0.3, ci_high=0.1, probability_better=0.8)


def test_a_candidate_better_on_average_is_served_provisionally() -> None:
    registry = Registry()
    servable = {"global_mean": Lookup(None, {}, 5.0)}

    note = promote_provisional(
        registry,
        "toy-y",
        "3",
        np.array([1.0, 1.0]),
        {"global_mean": np.array([2.0, 2.0])},  # type: ignore[arg-type]
        servable,
        pd.DataFrame({"slot": ["am"]}),
        UNSURE,
    )

    assert registry.aliases[CHAMPION] == "3"
    assert note.startswith(PROVISIONAL) and "80% sure" in note
    assert registry.tags["3"]["gate"] == note


def test_a_candidate_worse_than_a_baseline_leaves_the_baseline_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("toy")
    client = MlflowClient()
    x_test = pd.DataFrame({"slot": ["am", "pm"], "size": [1.0, 2.0]})
    slot_mean = Lookup("slot", {"am": 10.0, "pm": 4.0}, 7.0)

    with mlflow.start_run():
        note = promote_provisional(
            client,
            "toy-y",
            "1",
            np.array([3.0, 3.0]),
            {"global_mean": np.array([2.5, 2.5]), "slot_mean": np.array([1.0, 1.0])},
            {"global_mean": Lookup(None, {}, 7.0), "slot_mean": slot_mean},
            x_test,
            UNSURE,
        )

    assert has_champion(client, "toy-y")
    served = load_champion("toy-y", mlflow.get_tracking_uri(), tmp_path / "cache")
    assert served.model.predict(x_test).tolist() == [10.0, 4.0]
    assert served.gate == note and "the slot_mean baseline" in note and "v1" in note
    # The cache keeps the note: a restart without the registry still says it.
    cached = load_champion("toy-y", "sqlite:///nowhere.db", tmp_path / "cache", probe_retries=0)
    assert cached.source == "cache" and cached.gate == note
    assert not has_champion(client, "no-such-model")


def test_training_with_no_champion_always_leaves_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate no candidate can pass, and no champion yet: something is served anyway."""
    monkeypatch.chdir(tmp_path)
    model = ModelConfig(
        name="y_model",
        description="A toy target.",
        example={"slot": "am", "size": 1.0},
        items=ItemsConfig(table="rows", id="item_id", time="day", period="period"),
        spec=spec(),
        training=training(min_probability_better=1.01),
        target_bands=TargetBands(edges=[5.0], labels=["low", "high"]),
    )
    config = DomainConfig(
        name="toy",
        sources={},
        models=[model],
        analysis=AnalysisConfig(min_rows=1, permutation_repeats=2, published_figures=[]),
        monitoring=MonitoringConfig(drift_share=0.5),
    )
    write_table(rows(90), tmp_path / "features" / model.features_table, {})
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    result = train_model(config, "y_model", tmp_path, tracking_uri)

    assert not result.promoted and result.provisional.startswith(PROVISIONAL)
    champion = MlflowClient(tracking_uri).get_model_version_by_alias("toy-y", CHAMPION)
    assert champion.tags["gate"] == result.provisional
    # Trained again with the provisional one in place, the gate decides as before.
    again = train_model(config, "y_model", tmp_path, tracking_uri)
    assert not again.promoted and again.provisional == ""


def test_an_answer_from_a_provisional_model_says_so() -> None:
    provisional = described(
        {"target": "y", "prediction": 2.0, "model_gate": "provisional: the slot_mean baseline"}
    )
    promoted = described({"target": "y", "prediction": 2.0, "model_gate": "promoted"})

    assert provisional.endswith("[the model is provisional - provisional: the slot_mean baseline]")
    assert promoted == "y = 2.00"
