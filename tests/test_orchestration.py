"""Dagster is a thin layer: adding a domain must add a whole graph, and the asset
checks must fail loudly when the data breaks its contract."""

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from dagster import AssetKey, AssetSelection, build_sensor_context, materialize

from domains.coffee.adapter import CoffeeAdapter
from mlops_core.adapter import ApiExtraction
from mlops_core.config import Settings
from mlops_core.data.extract import CHECKS, MANIFEST_NAME, Check, Manifest
from mlops_core.ml.registry import NoChampion
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
        "tea/ico_prices_reads",
        "tea/roaster_catalogs_reads",
        "tea/profeco_prices_reads",
        "tea/review_features",
        "tea/review_model",
        "tea/review_predictions",
        "tea/review_drift",
        "tea/offer_features",
        "tea/offer_model",
        "tea/offer_predictions",
        "tea/offer_drift",
        "tea/green_price_features",
        "tea/green_price_model",
        "tea/green_price_predictions",
        "tea/green_price_drift",
    ]
    assert [j.name for j in defs.jobs] == [
        "coffee_data",
        "coffee_ml",
        "tea_data",
        "tea_ml",
        "coffee_ico_prices_reads",
        "coffee_roaster_catalogs_reads",
        "coffee_profeco_prices_reads",
        "tea_ico_prices_reads",
        "tea_roaster_catalogs_reads",
        "tea_profeco_prices_reads",
    ]


def test_without_a_list_every_installed_domain_gets_a_graph(tmp_path: Path) -> None:
    defs = build_definitions(settings=Settings(data_dir=tmp_path))

    assert [j.name for j in defs.jobs] == [
        "coffee_data",
        "coffee_ml",
        "coffee_ico_prices_reads",
        "coffee_roaster_catalogs_reads",
        "coffee_profeco_prices_reads",
    ]


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
        "monitor_model": Stub(None),
        "validate_raw": Stub({"cqi_2018": SimpleNamespace(frame=pl.DataFrame({"a": [1]}))}),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(definitions, name, stub)
    monkeypatch.setattr(definitions, "http_client", contextmanager(lambda: iter([None])))
    assets, _ = features_asset(tmp_path, coffee_adapter)
    # The pipelines' assets; the partitioned reads run a day at a time, on their own.
    reads = [["coffee", f"{name}_reads"] for name in coffee_adapter.config.accumulate]

    result = materialize(assets, selection=AssetSelection.all() - AssetSelection.assets(*reads))

    assert result.success
    # The data steps run once; each model step once per model.
    models = len(coffee_adapter.config.models)
    per_model = {"build_features", "train_model", "batch_predict", "monitor_model"}
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


def test_a_model_no_version_has_passed_is_recorded_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_champion(*args: object) -> Path:
        raise NoChampion("no champion and no cache")

    monkeypatch.setattr(definitions, "batch_predict", no_champion)
    defs = build_definitions(settings=Settings(data_dir=tmp_path))
    predictions = defs.resolve_assets_def(AssetKey(["coffee", "green_price_predictions"]))

    result = predictions()

    assert result.metadata == {"skipped": "no version has passed the gate"}


def test_the_drift_asset_records_the_monitors_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verdict = SimpleNamespace(
        current="cqi_2023", drifted_share=1.0, retrain=True, reasons=["the target drifted"]
    )
    monkeypatch.setattr(definitions, "monitor_model", lambda *args: verdict)
    defs = build_definitions(settings=Settings(data_dir=tmp_path))
    drift = defs.resolve_assets_def(AssetKey(["coffee", "review_drift"]))

    result = drift()

    assert result.metadata == {
        "current": "cqi_2023",
        "drifted_share": 1.0,
        "retrain": "True",
        "reasons": "the target drifted",
    }


def test_the_data_pipeline_runs_on_the_domains_schedule(tmp_path: Path) -> None:
    defs = build_definitions(settings=Settings(data_dir=tmp_path))

    (schedule,) = defs.schedules
    assert (schedule.name, schedule.job_name) == ("coffee_daily_data", "coffee_data")
    assert (schedule.cron_schedule, schedule.execution_timezone) == (
        "0 7 * * *",
        "America/Mexico_City",
    )


def sensor_named(tmp_path: Path, name: str) -> Any:
    defs = build_definitions(settings=Settings(data_dir=tmp_path))
    return next(s for s in defs.sensors if s.name == name)


def test_new_data_for_a_model_scores_and_monitors_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_new_data = sensor_named(tmp_path, "coffee_new_data")
    monkeypatch.setattr(definitions, "new_data", lambda adapter, data_dir: {"offer": "d4"})

    (request,) = on_new_data.evaluate_tick(build_sensor_context()).run_requests

    assert request.run_key == "offer:d4:score"  # one change, one run
    assert [key.to_user_string() for key in request.asset_selection] == [
        "coffee/offer_features",
        "coffee/offer_predictions",
        "coffee/offer_drift",
    ]
    monkeypatch.setattr(definitions, "new_data", lambda adapter, data_dir: {})
    skipped = on_new_data.evaluate_tick(build_sensor_context())
    assert skipped.run_requests == [] and "newest data" in skipped.skip_message


def test_a_due_retraining_trains_and_lets_the_gate_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_drift = sensor_named(tmp_path, "coffee_retrain")
    monkeypatch.setattr(definitions, "retraining_due", lambda config, data_dir: {"review": "d7"})

    (request,) = on_drift.evaluate_tick(build_sensor_context()).run_requests

    assert request.run_key == "review:d7:retrain"
    assert [key.to_user_string() for key in request.asset_selection] == [
        "coffee/review_model",
        "coffee/review_predictions",
        "coffee/review_drift",
    ]
    monkeypatch.setattr(definitions, "retraining_due", lambda config, data_dir: {})
    assert on_drift.evaluate_tick(build_sensor_context()).run_requests == []


# --- The reads of a source whose history is its downloads, a partition a day ---------------


def read_on(data_dir: Path, source: str, at: datetime) -> str:
    """A complete raw ingestion of `source` made at `at`, as extract left one before
    downloads were logged; its partition's name."""
    partition = data_dir / "coffee" / "raw" / source / f"ingested_at={at:%Y%m%dT%H%M%S%fZ}"
    partition.mkdir(parents=True)
    (partition / "read.json").write_text("{}", encoding="utf-8")
    manifest = Manifest(
        source=source, url="https://shop.test/", filename="read.json",
        sha256=at.isoformat(), size_bytes=2, ingested_at=at,
    )  # fmt: skip
    (partition / MANIFEST_NAME).write_text(manifest.model_dump_json(), encoding="utf-8")
    return partition.name


def reads_asset(tmp_path: Path, source: str) -> tuple[Any, Any]:
    defs = build_definitions(settings=Settings(data_dir=tmp_path))
    by_name = {a.key.path[-1]: a for a in defs.assets}
    return by_name[f"{source}_reads"], by_name["raw_sources"].to_source_asset()


def test_a_day_a_source_was_read_is_a_partition_checked_read_by_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two reads on 21 September in the city (one of them after 6 pm: the 22nd in UTC),
    # none on the 22nd, one on the 23rd that found what the 21st left.
    for at in ("2026-09-21T15:00:00", "2026-09-22T02:00:00"):
        last = read_on(tmp_path, "roaster_catalogs", datetime.fromisoformat(at).replace(tzinfo=UTC))
    unchanged = Check(
        checked_at=datetime(2026, 9, 23, 15, tzinfo=UTC), partition=last, changed=False
    )
    log = tmp_path / "coffee" / "raw" / "roaster_catalogs" / CHECKS
    log.write_text(unchanged.model_dump_json() + "\n", encoding="utf-8")
    checked: list[str] = []

    def validate_read(adapter: object, name: str, artifact: Any) -> Any:
        checked.append(artifact.partition.name)
        return SimpleNamespace(frame=pl.DataFrame({"offer": [1, 2]}))

    monkeypatch.setattr(definitions, "validate_read", validate_read)
    reads, raw = reads_asset(tmp_path, "roaster_catalogs")

    assert reads.partitions_def.get_partition_keys()[:3] == [
        "2026-09-21", "2026-09-22", "2026-09-23",
    ]  # fmt: skip
    assert reads.partitions_def.timezone == "America/Mexico_City"
    result = materialize([reads, raw], partition_key="2026-09-21")
    metadata = result.asset_materializations_for_node("coffee__roaster_catalogs_reads")[0].metadata
    assert (metadata["downloads"].value, metadata["new"].value, metadata["rows"].value) == (2, 2, 4)
    assert len(checked) == 2
    # Read on the 23rd and nothing new: what it showed is checked again, and said to be old.
    again = materialize([reads, raw], partition_key="2026-09-23")
    metadata = again.asset_materializations_for_node("coffee__roaster_catalogs_reads")[0].metadata
    assert (metadata["downloads"].value, metadata["new"].value) == (1, 0)
    assert checked[-1] == last
    unread = materialize([reads, raw], partition_key="2026-09-22", raise_on_error=False)
    assert not unread.success
    (failure,) = [e for e in unread.all_events if e.event_type_value == "STEP_FAILURE"]
    assert "was not read on 2026-09-22" in failure.event_specific_data.error.message


def test_each_new_read_asks_for_its_day_once(tmp_path: Path) -> None:
    on_reads = sensor_named(tmp_path, "coffee_reads")
    assert "no source" in on_reads.evaluate_tick(build_sensor_context()).skip_message
    read_on(tmp_path, "ico_prices", datetime(2026, 9, 26, 4, 13, tzinfo=UTC))
    read_on(tmp_path, "roaster_catalogs", datetime(2026, 9, 27, 11, 6, tzinfo=UTC))
    # Rebuilt: a source's partitions start the first day it was read.
    defs = build_definitions(settings=Settings(data_dir=tmp_path))
    on_reads = next(s for s in defs.sensors if s.name == "coffee_reads")

    context = build_sensor_context(definitions=defs)
    requests = on_reads.evaluate_tick(context).run_requests

    assert [(r.job_name, r.partition_key, r.run_key) for r in requests] == [
        ("coffee_ico_prices_reads", "2026-09-25",
         "ico_prices:2026-09-25:2026-09-26T04:13:00+00:00"),
        ("coffee_roaster_catalogs_reads", "2026-09-27",
         "roaster_catalogs:2026-09-27:2026-09-27T11:06:00+00:00"),
    ]  # fmt: skip


def test_a_domain_whose_sources_keep_no_history_gets_no_reads(
    coffee_adapter: CoffeeAdapter, tmp_path: Path
) -> None:
    latest_only = CoffeeAdapter(coffee_adapter.config.model_copy(update={"accumulate": []}))

    defs = build_definitions([latest_only], Settings(data_dir=tmp_path))

    assert not any(a.key.path[-1].endswith("_reads") for a in defs.assets)
    assert [s.name for s in defs.sensors] == ["coffee_new_data", "coffee_retrain"]


def test_nothing_starts_on_its_own_unless_the_deployment_says_so(tmp_path: Path) -> None:
    """By hand is the default: opening the UI must not start downloads or retraining."""
    from dagster import DefaultScheduleStatus, DefaultSensorStatus

    by_hand = build_definitions(settings=Settings(data_dir=tmp_path))
    automated = build_definitions(settings=Settings(data_dir=tmp_path, automate=True))

    assert {s.default_status for s in by_hand.schedules} == {DefaultScheduleStatus.STOPPED}
    assert {s.default_status for s in by_hand.sensors} == {DefaultSensorStatus.STOPPED}
    assert {s.default_status for s in automated.schedules} == {DefaultScheduleStatus.RUNNING}
    assert {s.default_status for s in automated.sensors} == {DefaultSensorStatus.RUNNING}
