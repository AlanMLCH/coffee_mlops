"""The registry is the source of truth; the cache is what keeps the API alive without it."""

from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyRegressor

from mlops_core.ml.registry import CACHED_METADATA, NoChampion, load_champion

UNREACHABLE = "http://127.0.0.1:1"  # nothing listens here


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A tracking server with one champion version registered."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test")
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            DummyRegressor().fit([[0.0]], [82.0]),
            name="model",
            registered_model_name="m",
            pip_requirements=["scikit-learn"],  # skip slow environment inference
        )
    MlflowClient(uri).set_registered_model_alias("m", "champion", info.registered_model_version)
    return uri


def test_champion_is_loaded_and_cached(registry: str, tmp_path: Path) -> None:
    cache = tmp_path / "cache"

    served = load_champion("m", registry, cache)

    assert (served.version, served.source) == ("1", "registry")
    assert (cache / "m" / CACHED_METADATA).is_file()  # one folder per registered model


def test_cache_serves_when_the_registry_is_unreachable(registry: str, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    load_champion("m", registry, cache)

    served = load_champion("m", UNREACHABLE, cache)

    assert (served.version, served.source) == ("1", "cache")
    assert served.model.predict([[0.0]])[0] == 82.0


def test_no_registry_and_no_cache_fails_with_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(NoChampion, match="`gate` tag"):
        load_champion("m", UNREACHABLE, tmp_path / "cache")


def test_a_half_copied_cache_is_not_served(registry: str, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    load_champion("m", registry, cache)
    (cache / "m" / CACHED_METADATA).unlink()  # metadata is written last

    with pytest.raises(FileNotFoundError):
        load_champion("m", UNREACHABLE, cache)


def test_each_registered_model_keeps_its_own_copy(registry: str, tmp_path: Path) -> None:
    """A domain with two models shares one cache directory: neither may overwrite the other."""
    cache = tmp_path / "cache"
    load_champion("m", registry, cache)

    with pytest.raises(FileNotFoundError):
        load_champion("other", UNREACHABLE, cache)  # m's copy is not other's
