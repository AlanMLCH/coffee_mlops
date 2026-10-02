"""`mlops status`: what is ready and what is left to do, each with the command that makes
it ready - checked on a raw layer built from the fixtures, and on services stood in for."""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.config import CoffeeCredentials
from mlops_core import cli
from mlops_core import status as checks
from mlops_core.config import Settings
from mlops_core.data.clean import build_clean
from mlops_core.status import Finding, ago, stamp, to_do
from mlops_core.storage import write_table

FIXED = datetime(2026, 9, 29, 12, tzinfo=UTC)


def found(findings: list[Finding], name: str) -> Finding:
    return next(f for f in findings if f.name == name)


def test_a_fresh_raw_layer_asks_for_what_is_built_from_it(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    findings = checks.data_findings(coffee_adapter, raw_dir.parent)

    assert found(findings, "psd_coffee").ready is True
    assert found(findings, "ico_prices").detail.endswith("1 read kept")  # accumulates
    documents = found(findings, "documents")
    # The fixtures serve the documents a publisher serves; the others are handed over, and
    # a document only a person can fetch is optional: a note, not a task.
    assert documents.ready is None and "put " in documents.detail and "inbox" in documents.detail
    clean = found(findings, "clean")
    assert (clean.ready, clean.fix) == (False, "make clean-layer")
    assert found(findings, "review_features").fix == "make ml"
    assert found(findings, "analysis").fix == "make analysis"


def test_a_source_past_its_refresh_is_due_and_one_never_read_is_to_download(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    (raw_dir / "psd_coffee" / "checked_at").write_text(
        (FIXED - timedelta(days=2)).isoformat(), encoding="utf-8"
    )
    for partition in (raw_dir / "fred_usd_mxn").glob("*"):
        for file in partition.glob("*") if partition.is_dir() else []:
            file.unlink()
        partition.rmdir() if partition.is_dir() else partition.unlink()

    findings = checks.data_findings(coffee_adapter, raw_dir.parent, now=FIXED)

    psd = found(findings, "psd_coffee")  # refreshed every 24 hours
    assert (psd.ready, psd.fix) == (False, "make extract")
    assert "2 days ago - due again" in psd.detail
    fred = found(findings, "fred_usd_mxn")
    assert (fred.ready, fred.detail, fred.fix) == (False, "never downloaded", "make extract")


def test_a_layer_built_before_the_newest_download_is_stale(
    coffee_adapter: CoffeeAdapter, raw_dir: Path
) -> None:
    data_dir = raw_dir.parent
    build_clean(coffee_adapter, data_dir)

    built = found(checks.data_findings(coffee_adapter, data_dir), "clean")
    (raw_dir / "psd_coffee" / "ingested_at=20991231T000000000000Z").mkdir()
    stale = found(checks.data_findings(coffee_adapter, data_dir), "clean")

    assert built.ready is True and "tables, built" in built.detail
    assert (stale.ready, stale.detail) == (False, "built before the newest download")


def test_a_model_the_gate_never_promoted_is_a_note_and_a_due_retraining_is_to_do(
    coffee_adapter: CoffeeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mlops_core.orchestration.triggers.retraining_due",
        lambda config, data_dir: {"review": "0123456789abcdef"},
    )
    champions = {"coffee-total-cup-points": "7", "coffee-price-per-kg": "5"}

    findings = checks.model_findings(coffee_adapter.config, tmp_path, champions.get)

    assert found(findings, "review") == Finding("models", "review", True, "champion v7")
    assert found(findings, "green_price").ready is None
    due = found(findings, "review retraining")
    assert (due.ready, due.fix) == (False, "make retrain") and "0123456789ab," in due.detail


def services(
    tags: list[str], api: dict[str, Any] | None, down: bool = False
) -> Callable[[httpx.Request], httpx.Response]:
    def answer(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in tags]})
        if request.url.port == 8000:
            return httpx.Response(200, json=api)
        return httpx.Response(200, text="OK")

    return answer


def test_services_say_what_they_hold_and_what_starts_them(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = raw_dir.parent
    build_clean(coffee_adapter, data_dir)
    monkeypatch.setattr("mlops_core.rag.vectors.index_metadata",
                        lambda client, domain: {"chunks_digest": "other"})  # fmt: skip
    api = {"status": "partial", "models": {"review": "1", "green_price": None}}
    http = httpx.Client(transport=httpx.MockTransport(services(["qwen3.5:4b"], api)))

    findings = checks.service_findings(
        Settings(mlflow_tracking_uri="http://mlflow.test"), coffee_adapter.config, data_dir, http
    )

    assert found(findings, "MLflow").ready is True
    assert found(findings, "prediction API").detail == "review v1; not served: green_price"
    assert found(findings, "index").fix == "make index"  # built from other chunks
    assert found(findings, "qwen3.5:4b").ready is True
    assert found(findings, "qwen3-embedding:0.6b").fix == "ollama pull qwen3-embedding:0.6b"


def test_services_that_are_down_are_findings_not_crashes(
    coffee_adapter: CoffeeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable(client: object, domain: str) -> dict[str, Any]:
        raise ConnectionError("Connection refused")

    monkeypatch.setattr("mlops_core.rag.vectors.index_metadata", unreachable)
    http = httpx.Client(transport=httpx.MockTransport(services([], None, down=True)))
    settings = Settings(mlflow_tracking_uri="sqlite:///registry.db")

    findings = checks.service_findings(settings, coffee_adapter.config, tmp_path, http)

    registry = found(findings, "MLflow")
    assert (registry.ready, registry.detail) == (True, "a local registry: sqlite:///registry.db")
    assert found(findings, "prediction API").fix == "make services-up PROFILE=api"
    assert found(findings, "index").fix == "make clean-layer"  # no chunks to index yet
    assert found(findings, "local models").fix == "start Ollama"
    (tmp_path / "clean").mkdir()
    build = checks._index(settings, coffee_adapter.config, tmp_path)
    assert build.ready is False


def test_the_index_is_current_when_it_was_built_from_the_chunks_on_disk(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlops_core.config import CHUNKS_TABLE
    from mlops_core.rag.vectors import chunks_digest
    from mlops_core.storage import read_table

    data_dir = raw_dir.parent
    build_clean(coffee_adapter, data_dir)
    digest = chunks_digest(read_table(data_dir / "clean" / CHUNKS_TABLE))
    monkeypatch.setattr("mlops_core.rag.vectors.index_metadata",
                        lambda client, domain: {"chunks_digest": digest})  # fmt: skip

    index = checks._index(Settings(), coffee_adapter.config, data_dir)
    no_documents = coffee_adapter.config.model_copy(update={"documents": []})

    assert index.ready is True and index.detail.startswith("current:")
    assert checks._index(Settings(), no_documents, data_dir).ready is None


def test_an_unreachable_index_or_registry_says_which_service_to_start(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable(client: object, domain: str) -> dict[str, Any]:
        raise ConnectionError("Connection refused")

    data_dir = raw_dir.parent
    build_clean(coffee_adapter, data_dir)
    monkeypatch.setattr("mlops_core.rag.vectors.index_metadata", unreachable)
    http = httpx.Client(transport=httpx.MockTransport(services([], None, down=True)))

    findings = checks.service_findings(
        Settings(mlflow_tracking_uri="http://mlflow.test"), coffee_adapter.config, data_dir, http
    )

    index, registry = found(findings, "index"), found(findings, "MLflow")
    assert (index.detail, index.fix) == (
        "Connection refused",
        "make services-up PROFILE=ai, make index",
    )
    assert (registry.ready, registry.fix) == (False, "make services-up PROFILE=ml")


def test_without_the_extras_the_services_they_need_are_notes(
    coffee_adapter: CoffeeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "qdrant_client", None)  # an import of it now fails
    monkeypatch.setitem(sys.modules, "mlops_core.agent.graph", None)
    http = httpx.Client(transport=httpx.MockTransport(services([], None)))

    index = checks._index(Settings(), coffee_adapter.config, tmp_path)
    models = checks._local_models(Settings(), http)

    assert (index.ready, index.detail) == (None, "the rag extra is not installed")
    assert [(m.ready, m.detail) for m in models] == [(None, "the agent extra is not installed")]


def test_features_built_before_their_data_and_an_analysis_before_its_layer(
    coffee_adapter: CoffeeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = coffee_adapter.config
    frame = pl.DataFrame({"x": [1]})
    for model in config.models:
        write_table(frame, tmp_path / "features" / model.features_table, {})
        write_table(frame, tmp_path / "predictions" / model.predictions_table, {})
    write_table(frame, tmp_path / "analysis" / "study", {}, at=FIXED)
    write_table(frame, tmp_path / "clean" / "items", {}, at=FIXED + timedelta(hours=1))
    monkeypatch.setattr("mlops_core.orchestration.triggers.new_data",
                        lambda adapter, data_dir: {"offer": "v2"})  # fmt: skip

    stale = checks._models_layers(coffee_adapter, tmp_path)
    write_table(frame, tmp_path / "analysis" / "study", {}, at=FIXED + timedelta(hours=2))
    current = found(checks._models_layers(coffee_adapter, tmp_path), "analysis")

    assert found(stale, "offer_features").detail == "older than its data"
    assert found(stale, "review_features").ready is True
    assert not [f for f in stale if f.name.endswith("_predictions")]  # all three scored
    assert found(stale, "analysis").detail == "run before the last clean build"
    assert (current.ready, current.detail) == (True, "run 2026-09-29 14:00")


def test_the_champion_is_read_from_the_registry_by_its_alias(tmp_path: Path) -> None:
    from mlflow import MlflowClient

    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    champion = cli._champion_version(Settings(mlflow_tracking_uri=uri))
    registry = MlflowClient(tracking_uri=uri, registry_uri=uri)
    registry.create_registered_model("promoted")
    registry.create_model_version("promoted", source=(tmp_path / "model").as_uri())
    registry.set_registered_model_alias("promoted", "champion", "1")
    registry.create_registered_model("never promoted")

    assert champion("promoted") == "1"
    assert champion("never promoted") is None
    assert champion("unknown") is None


def test_keys_are_said_set_or_missing_never_shown(coffee_adapter: CoffeeAdapter) -> None:
    # Every key named: a developer's .env would fill in the ones left out.
    keys = CoffeeCredentials(
        denue_token=SecretStr("secret-token"), usda_fas_api_key=None, inpc_token=None
    )

    findings = checks.key_findings(CoffeeAdapter(coffee_adapter.config, keys))

    assert [(f.ready, f.fix) for f in findings] == [(True, ""), (None, ""), (None, "")]
    assert findings[1].detail.startswith("missing: optional")
    assert "secret-token" not in repr(findings)


def test_without_a_key_an_api_source_never_read_is_a_note_and_a_file_is_still_to_do(
    coffee_adapter: CoffeeAdapter, tmp_path: Path
) -> None:
    keyless = CoffeeAdapter(coffee_adapter.config, CoffeeCredentials(denue_token=None))
    (tmp_path / "raw").mkdir()

    findings = checks.data_findings(keyless, tmp_path)

    assert found(findings, "denue_cafes").ready is None  # skipped at extract, said there
    assert found(findings, "psd_coffee").fix == "make extract"
    documents = found(findings, "documents")
    assert (documents.ready, documents.fix) == (False, "make extract")  # some can be fetched


def test_what_to_do_comes_in_the_order_it_should_run() -> None:
    findings = [
        Finding("data", "analysis", False, "", "make analysis"),
        Finding("data", "clean", False, "", "make clean-layer"),
        Finding("data", "osm", False, "", "make extract"),
        Finding("data", "fas", False, "", "make extract"),
        Finding("keys", "token", False, "", "set it in .env (see .env.example)"),
        Finding("models", "green_price", None, "no version has passed the gate"),
    ]

    assert to_do(findings) == {
        "set it in .env (see .env.example)": ["token"],
        "make extract": ["osm", "fas"],
        "make clean-layer": ["clean"],
        "make analysis": ["analysis"],
    }


def test_times_read_as_words() -> None:
    assert [ago(timedelta(days=2, hours=3)), ago(timedelta(hours=1)), ago(timedelta(seconds=5))] \
        == ["2 days", "1 hour", "moments"]  # fmt: skip
    assert stamp(Path("ingested_at=20260929T044001996244Z")) == datetime(
        2026, 9, 29, 4, 40, 1, 996244, tzinfo=UTC
    )


def test_the_status_command_prints_each_section_and_what_to_run(
    coffee_adapter: CoffeeAdapter, raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def stood_in() -> Iterator[httpx.Client]:
        api = {"status": "ok", "models": {"review": "1"}}
        yield httpx.Client(transport=httpx.MockTransport(services(["qwen3.5:4b"], api)))

    monkeypatch.setattr(cli, "http_client", stood_in)
    monkeypatch.setattr(cli, "load_adapter", lambda domain: coffee_adapter)
    monkeypatch.setattr(
        cli, "_champion_version", lambda settings: {"coffee-total-cup-points": "1"}.get
    )
    monkeypatch.setattr("mlops_core.orchestration.triggers.retraining_due", lambda c, d: {})
    monkeypatch.setattr("mlops_core.rag.vectors.index_metadata", lambda client, domain: {})
    monkeypatch.setenv("MLOPS_DATA_DIR", str(raw_dir.parent.parent))

    result = CliRunner().invoke(cli.app, ["status"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "keys"
    assert "  [ok]    review: champion v1" in result.output
    assert "To do, in this order:" in result.output
    assert "  make clean-layer   (clean, index)" in result.output  # no chunks to index yet


def test_with_the_registry_down_the_champions_are_unknown(
    coffee_adapter: CoffeeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def down() -> Iterator[httpx.Client]:
        yield httpx.Client(transport=httpx.MockTransport(services([], None, down=True)))

    monkeypatch.setattr(cli, "http_client", down)
    monkeypatch.setattr(cli, "load_adapter", lambda domain: coffee_adapter)
    monkeypatch.setattr("mlops_core.rag.vectors.index_metadata", lambda client, domain: {})
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MLOPS_MLFLOW_TRACKING_URI", "http://mlflow.test")

    result = CliRunner().invoke(cli.app, ["status"])

    assert result.exit_code == 0, result.output
    assert "[note]  champions: unknown: MLflow is down" in result.output
    assert "make services-up PROFILE=ml" in result.output
