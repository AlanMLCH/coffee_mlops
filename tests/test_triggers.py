"""When the orchestrator acts on its own: new data for a model, and a retraining that is
due and not done. Built from partitions and runs written here, with a toy model."""

from pathlib import Path
from types import SimpleNamespace

import mlflow
import polars as pl

from mlops_core.config import DomainConfig
from mlops_core.monitoring.drift import MONITORING, VERDICT_FILE, Verdict
from mlops_core.orchestration.triggers import new_data, retraining_due
from mlops_core.storage import write_table
from tests.test_monitoring import model


def toy(coffee_config: DomainConfig) -> SimpleNamespace:
    """An adapter whose one model reads `lots` and, as context, `prices`."""
    config = coffee_config.model_copy(update={"models": [model()]})
    return SimpleNamespace(config=config, context_tables=lambda name: ("prices",))


def clean(data_dir: Path, table: str, raw: str) -> str:
    path = write_table(pl.DataFrame({"v": [1]}), data_dir / "clean" / table, {"src": raw})
    return path.parent.name


def test_a_model_has_new_data_only_when_the_raw_data_behind_it_changed(
    tmp_path: Path, coffee_config: DomainConfig
) -> None:
    adapter = toy(coffee_config)
    assert new_data(adapter, tmp_path) == {}  # no clean layer yet: nothing to build from

    lots, prices = clean(tmp_path, "lots", "raw=A"), clean(tmp_path, "prices", "raw=P")
    first = new_data(adapter, tmp_path)
    assert list(first) == ["price"]  # features never built

    write_table(pl.DataFrame({"v": [1]}), tmp_path / "features" / "price_features",
                {"lots": lots, "prices": prices})  # fmt: skip
    assert new_data(adapter, tmp_path) == {}
    clean(tmp_path, "lots", "raw=A")  # rebuilt, same data: nothing new
    assert new_data(adapter, tmp_path) == {}
    clean(tmp_path, "lots", "raw=B")  # a download brought something new
    changed = new_data(adapter, tmp_path)
    assert list(changed) == ["price"] and changed != first


def verdict(data_dir: Path, retrain: bool, version: str) -> None:
    table = write_table(pl.DataFrame({"v": [1]}), data_dir / MONITORING / "price_drift", {})
    decided = Verdict(model="price", current="2026", retrain=retrain, reasons=[],
                      data_version=version)  # fmt: skip
    (table.parent / VERDICT_FILE).write_text(decided.model_dump_json(), encoding="utf-8")


def test_a_retraining_is_due_once_per_data_version(
    tmp_path: Path, coffee_config: DomainConfig
) -> None:
    config = toy(coffee_config).config
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    assert retraining_due(config, tmp_path) == {}  # never monitored

    verdict(tmp_path, retrain=False, version="v1")
    assert retraining_due(config, tmp_path) == {}  # nothing called for

    verdict(tmp_path, retrain=True, version="v1")
    assert retraining_due(config, tmp_path) == {"price": "v1"}

    mlflow.set_experiment("coffee-price")  # the model's own experiment
    with mlflow.start_run():
        mlflow.set_tag("data_version", "v1")
    assert retraining_due(config, tmp_path) == {}  # trained on it already

    verdict(tmp_path, retrain=True, version="v2")
    assert retraining_due(config, tmp_path) == {"price": "v2"}
