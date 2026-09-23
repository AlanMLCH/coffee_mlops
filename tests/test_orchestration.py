"""Dagster is a thin layer: adding a domain must add a whole graph, and the asset
checks must fail loudly when the data breaks its contract."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from dagster import AssetSelection, materialize

from domains.coffee.adapter import CoffeeAdapter
from mlops_core.adapter import ApiExtraction
from mlops_core.config import Settings
from mlops_core.ml.train import TrainResult
from mlops_core.orchestration import definitions
from mlops_core.orchestration.definitions import build_definitions
from mlops_core.storage import write_table

FEATURES_TABLE = "review_features"


@pytest.fixture
def two_domains(coffee_adapter: CoffeeAdapter) -> list[CoffeeAdapter]:
    """Coffee, and the same adapter answering to another name."""
    tea = CoffeeAdapter(coffee_adapter.config.model_copy(update={"name": "tea"}))
    return [coffee_adapter, tea]


def test_each_domain_adds_its_own_graph(two_domains: list[CoffeeAdapter], tmp_path: Path) -> None:
    defs = build_definitions(two_domains, Settings(data_dir=tmp_path))

    assert [a.key.to_user_string() for a in defs.assets if a.key.path[0] == "tea"] == [
        "tea/raw_sources",
        "tea/clean_tables",
        "tea/review_features",
        "tea/review_model",
        "tea/review_predictions",
        "tea/offer_features",
        "tea/offer_model",
        "tea/offer_predictions",
    ]
    assert [j.name for j in defs.jobs] == ["coffee_data", "coffee_ml", "tea_data", "tea_ml"]


def test_without_a_list_every_installed_domain_gets_a_graph(tmp_path: Path) -> None:
    defs = build_definitions(settings=Settings(data_dir=tmp_path))

    assert [j.name for j in defs.jobs] == ["coffee_data", "coffee_ml"]


def features_asset(tmp_path: Path, adapter: CoffeeAdapter | None = None) -> tuple[list, object]:
    """Every asset and check, plus the key of the feature table asset."""
    adapters = [adapter] if adapter else None
    defs = build_definitions(adapters, settings=Settings(data_dir=tmp_path))
    assets = [*defs.assets, *(defs.asset_checks or [])]
    key = next(a.key for a in defs.assets if a.key.path[-1] == FEATURES_TABLE)
    return assets, key


def write_features(
    tmp_path: Path, extra: dict[str, list[float]], table: str = FEATURES_TABLE
) -> None:
    frame = pl.DataFrame({"item_id": ["a"], "target": [83.0], **extra})
    write_table(frame, tmp_path / "coffee" / "features" / table, inputs={})


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
    monkeypatch.setattr(definitions, "build_features", lambda *_: Path("written"))
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, coffee_adapter: CoffeeAdapter
) -> None:
    for model in coffee_adapter.config.models:
        write_features(tmp_path, {}, model.features_table)
    artifact = SimpleNamespace(manifest=SimpleNamespace(size_bytes=10))
    # The orchestrator pulls the API sources too, through the domain's adapter, or DENUE
    # and OSM would arrive only when someone typed the command.
    extract = Stub(
        ApiExtraction(artifacts={"osm_places": artifact}, skipped={"denue_cafes": "no token"})
    )
    monkeypatch.setattr(coffee_adapter, "extract", extract)
    stubs = {
        "extract_all": Stub({"cqi_2018": artifact}),
        "fetch_documents": Stub(({"wcr_arabica_catalog": artifact}, {"sca_103_descriptive": "x"})),
        "build_clean": Stub({"coffee_reviews": Path("reviews.parquet")}),
        "build_features": Stub(Path("features.parquet")),
        "train_model": Stub(TrainResult("run-1", "3", True, {"test_mae": 1.5})),
        "batch_predict": Stub(Path("predictions.parquet")),
        "validate_raw": Stub({"cqi_2018": SimpleNamespace(frame=pl.DataFrame({"a": [1]}))}),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(definitions, name, stub)
    monkeypatch.setattr(definitions, "http_client", contextmanager(lambda: iter([None])))
    assets, _ = features_asset(tmp_path, coffee_adapter)

    result = materialize(assets)

    assert result.success
    # The data steps run once; each model step once per model.
    models = len(coffee_adapter.config.models)
    per_model = {"build_features", "train_model", "batch_predict"}
    assert {name: stub.calls for name, stub in stubs.items()} == {
        name: models if name in per_model else 1 for name in stubs
    }
    assert extract.calls == 1
    model = result.asset_materializations_for_node("coffee__review_model")[0]
    assert model.metadata["version"].value == "3"
    assert model.metadata["promoted"].value == "True"
    raw = result.asset_materializations_for_node("coffee__raw_sources")[0]
    assert raw.metadata["sources"].value == 3  # a file source, an API source, a document
    assert raw.metadata["skipped"].value == "denue_cafes (no token), sca_103_descriptive (x)"
