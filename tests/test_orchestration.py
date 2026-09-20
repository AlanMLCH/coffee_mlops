"""Dagster is a thin layer: adding a domain config must add a whole graph, and the
asset checks must fail loudly when the data breaks its contract."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from dagster import AssetSelection, materialize

from coffee_mlops.config import Settings
from coffee_mlops.data.sources import ApiExtraction
from coffee_mlops.ml.train import TrainResult
from coffee_mlops.orchestration import definitions
from coffee_mlops.orchestration.definitions import build_definitions
from coffee_mlops.storage import write_table

FEATURES_TABLE = "review_features"


@pytest.fixture
def two_domains(tmp_path: Path) -> Path:
    """A configs dir where the coffee config has been copied under another name."""
    configs = tmp_path / "configs"
    configs.mkdir()
    coffee = (Path("configs") / "coffee.yaml").read_text(encoding="utf-8")
    (configs / "coffee.yaml").write_text(coffee, encoding="utf-8")
    (configs / "tea.yaml").write_text(coffee.replace("name: coffee", "name: tea", 1), "utf-8")
    return configs


def test_each_domain_config_adds_its_own_graph(two_domains: Path, tmp_path: Path) -> None:
    defs = build_definitions(two_domains, Settings(data_dir=tmp_path))

    assert [a.key.to_user_string() for a in defs.assets if a.key.path[0] == "tea"] == [
        "tea/raw_sources",
        "tea/clean_tables",
        "tea/review_features",
        "tea/trained_model",
        "tea/review_predictions",
    ]
    assert [j.name for j in defs.jobs] == ["coffee_data", "coffee_ml", "tea_data", "tea_ml"]


def features_asset(tmp_path: Path) -> tuple[list, object]:
    """Every asset and check, plus the key of the feature table asset."""
    defs = build_definitions(settings=Settings(data_dir=tmp_path))
    assets = [*defs.assets, *(defs.asset_checks or [])]
    key = next(a.key for a in defs.assets if a.key.path[-1] == FEATURES_TABLE)
    return assets, key


def write_features(tmp_path: Path, extra: dict[str, list[float]]) -> None:
    table = pl.DataFrame({"review_id": ["a"], "total_cup_points": [83.0], **extra})
    write_table(table, tmp_path / "coffee" / "features" / FEATURES_TABLE, inputs={})


@pytest.mark.parametrize(
    ("extra", "passes"),
    [
        pytest.param({}, True, id="clean-feature-table"),
        pytest.param({"aroma": [8.0]}, False, id="sensory-score-leaked-in"),
    ],
)
def test_the_leakage_check_guards_the_feature_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, list[float]], passes: bool
) -> None:
    write_features(tmp_path, extra)
    monkeypatch.setattr(definitions, "build_features", lambda config, data_dir: Path("written"))
    assets, key = features_asset(tmp_path)

    result = materialize(assets, selection=AssetSelection.assets(key))

    checks = result.get_asset_check_evaluations()
    assert [check.passed for check in checks] == [passes]


class Stub:
    """Records that the asset called it, and stands in for the real step."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self.result


def test_every_asset_runs_its_own_pipeline_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_features(tmp_path, {})
    artifact = SimpleNamespace(manifest=SimpleNamespace(size_bytes=10))
    stubs = {
        "extract_all": Stub({"cqi_2018": artifact}),
        # The orchestrator pulls the API sources too, or DENUE and OSM would arrive
        # only when someone typed the command.
        "extract_api_sources": Stub(
            ApiExtraction(artifacts={"osm_cafes": artifact}, skipped={"denue_cafes": "no token"})
        ),
        "build_clean": Stub({"coffee_reviews": Path("reviews.parquet")}),
        "build_features": Stub(Path("features.parquet")),
        "train_model": Stub(TrainResult("run-1", "3", True, {"test_mae": 1.5})),
        "batch_predict": Stub(Path("predictions.parquet")),
        "validate_raw": Stub({"cqi_2018": SimpleNamespace(frame=pl.DataFrame({"a": [1]}))}),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(definitions, name, stub)
    monkeypatch.setattr(definitions, "http_client", contextmanager(lambda: iter([None])))
    assets, _ = features_asset(tmp_path)

    result = materialize(assets)

    assert result.success
    assert {name: stub.calls for name, stub in stubs.items()} == dict.fromkeys(stubs, 1)
    model = result.asset_materializations_for_node("coffee__trained_model")[0]
    assert model.metadata["version"].value == "3"
    assert model.metadata["promoted"].value == "True"
    raw = result.asset_materializations_for_node("coffee__raw_sources")[0]
    assert raw.metadata["sources"].value == 2  # one file source, one API source
    assert raw.metadata["skipped"].value == "denue_cafes (no token)"
