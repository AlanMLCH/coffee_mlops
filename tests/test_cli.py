import json
import logging
import os
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest
import typer
from typer.testing import CliRunner

from mlops_core import cli
from mlops_core.config import load_domain_config
from mlops_core.data.extract import http_client
from mlops_core.ml.registry import ServedModel
from mlops_core.ml.train import TrainResult
from mlops_core.storage import write_table
from tests.fakes import ConstantModel, RecordedServer, without_rate_limits


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: RecordedServer) -> Path:
    """Point the CLI at a temp data dir and at the recorded server instead of the internet."""

    @contextmanager
    def recorded_client() -> Iterator[httpx.Client]:
        with http_client(httpx.MockTransport(server.handler)) as client:
            yield client

    monkeypatch.setattr(cli, "http_client", recorded_client)
    fast = without_rate_limits(load_domain_config("coffee"))
    monkeypatch.setattr(cli, "load_domain_config", lambda domain: fast)
    monkeypatch.setenv("COFFEE_DATA_DIR", str(tmp_path))
    return tmp_path


def test_extract_writes_raw_layer_under_the_domain(data_dir: Path) -> None:
    result = CliRunner().invoke(cli.app, ["data", "extract", "--domain", "coffee"])

    assert result.exit_code == 0, result.output
    raw = data_dir / "coffee" / "raw"
    # The file sources and the API sources that need no credential.
    assert {p.name for p in raw.iterdir()} == {
        "cqi_2018",
        "cqi_2023",
        "psd_coffee",
        "cdmx_boroughs",
        "osm_cafes",
    }


def test_validate_runs_after_extract(data_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli.app, ["data", "extract"])

    result = runner.invoke(cli.app, ["data", "validate", "--domain", "coffee"])

    assert result.exit_code == 0, result.output
    assert "psd_coffee: 114 rows valid" in result.output


def test_clean_builds_the_clean_layer(data_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli.app, ["data", "extract"])

    result = runner.invoke(cli.app, ["data", "clean", "--domain", "coffee"])

    assert result.exit_code == 0, result.output
    clean = data_dir / "coffee" / "clean"
    assert {p.name for p in clean.iterdir()} == {
        "coffee_reviews",
        "market_context",
        "boroughs",
        "coffee_shops",
    }


def test_features_and_sql_run_on_the_built_layers(data_dir: Path) -> None:
    runner = CliRunner()
    for group, step in (("data", "extract"), ("data", "clean"), ("ml", "features")):
        assert runner.invoke(cli.app, [group, step]).exit_code == 0

    result = runner.invoke(cli.app, ["sql", "SELECT count(*) AS n FROM features.review_features"])

    assert result.exit_code == 0, result.output
    assert "25" in result.output


def test_data_run_chains_the_whole_etl(data_dir: Path) -> None:
    result = CliRunner().invoke(cli.app, ["data", "run"])

    assert result.exit_code == 0, result.output
    assert {p.name for p in (data_dir / "coffee" / "clean").iterdir()} == {
        "coffee_reviews",
        "market_context",
        "boroughs",
        "coffee_shops",
    }


def test_ml_run_chains_features_and_training(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    CliRunner().invoke(cli.app, ["data", "run"])
    monkeypatch.setattr(
        "mlops_core.ml.train.train_model",
        lambda config, data, uri: TrainResult("run-1", "1", False, {"test_mae": 2.0}),
    )
    monkeypatch.setattr(
        "mlops_core.ml.predict.load_champion",
        lambda *args, **kwargs: ServedModel(ConstantModel(), "1", "registry"),
    )

    result = CliRunner().invoke(cli.app, ["ml", "run"])

    assert result.exit_code == 0, result.output
    assert (data_dir / "coffee" / "features" / "review_features").is_dir()
    assert (data_dir / "coffee" / "predictions" / "review_predictions").is_dir()
    assert "not promoted" in result.output


def test_train_reports_version_and_gate_decision(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_train_model(config: object, data: Path, tracking_uri: str) -> TrainResult:
        calls.append((data, tracking_uri))
        return TrainResult("run-1", "3", True, {"test_mae": 1.5})

    monkeypatch.setattr("mlops_core.ml.train.train_model", fake_train_model)
    monkeypatch.setenv("COFFEE_MLFLOW_TRACKING_URI", "sqlite:///somewhere.db")

    result = CliRunner().invoke(cli.app, ["ml", "train"])

    assert result.exit_code == 0, result.output
    assert calls == [(data_dir / "coffee", "sqlite:///somewhere.db")]
    assert "v3: promoted to champion" in result.output


def test_missing_extra_reports_how_to_install_it(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit), cli._needs_extra("ml"):
        raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")

    assert "uv sync --extra ml" in capsys.readouterr().err


def test_request_urls_are_not_logged(data_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Signed URLs (and, from stage 2, API tokens) travel in URLs; they must not hit logs."""
    caplog.set_level(logging.DEBUG)

    CliRunner().invoke(cli.app, ["data", "extract"])

    assert "Signature" not in caplog.text


def test_extract_pulls_denue_when_the_token_is_there(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COFFEE_DENUE_TOKEN", "super-secret-token")

    result = CliRunner().invoke(cli.app, ["data", "extract"])

    assert result.exit_code == 0, result.output
    assert "denue_cafes:" in result.output
    inventory = data_dir / "coffee" / "raw" / "denue_cafes"
    assert inventory.is_dir()
    stored = json.loads(next(inventory.rglob("denue_cafes.json")).read_text(encoding="utf-8"))
    assert len(stored) == 3


def test_extract_pulls_openstreetmap_without_any_credential(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OSM is the geolocated source a public repository can actually keep: no key to
    configure, and a licence that allows storing what comes back."""
    monkeypatch.delenv("COFFEE_DENUE_TOKEN", raising=False)

    result = CliRunner().invoke(cli.app, ["data", "extract"])

    assert result.exit_code == 0, result.output
    assert "osm_cafes:" in result.output
    stored = json.loads(
        next((data_dir / "coffee" / "raw" / "osm_cafes").rglob("osm_cafes.json")).read_text(
            encoding="utf-8"
        )
    )
    assert len(stored["elements"]) == 5
    assert "ODbL" in stored["license"]


def test_extract_pulls_the_fas_balance_when_the_key_is_there(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COFFEE_USDA_FAS_API_KEY", "super-secret-key")

    result = CliRunner().invoke(cli.app, ["data", "extract"])

    assert result.exit_code == 0, result.output
    stored = next((data_dir / "coffee" / "raw" / "fas_psd_coffee").rglob("fas_psd_coffee.json"))
    assert len(json.loads(stored.read_text(encoding="utf-8"))["rows"]) == 114
    # A header, not a URL: the key reaches neither the manifest nor a cache file name.
    assert "super-secret-key" not in (stored.parent / "manifest.json").read_text()
    assert all("super-secret-key" not in p.name for p in (data_dir / "coffee").rglob("*"))


def test_extract_says_when_it_skips_a_source_for_want_of_a_credential(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh clone with no credentials must still build the whole of stage 1."""
    monkeypatch.delenv("COFFEE_DENUE_TOKEN", raising=False)
    monkeypatch.delenv("COFFEE_USDA_FAS_API_KEY", raising=False)

    result = CliRunner().invoke(cli.app, ["data", "extract"])

    assert result.exit_code == 0, result.output
    assert "denue_cafes: skipped, COFFEE_DENUE_TOKEN is not set" in result.output
    assert "fas_psd_coffee: skipped, COFFEE_USDA_FAS_API_KEY is not set" in result.output
    assert not (data_dir / "coffee" / "raw" / "denue_cafes").exists()
    assert "cqi_2018:" in result.output  # the file sources still ran


def test_prune_reports_what_it_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COFFEE_DATA_DIR", str(tmp_path))
    table = tmp_path / "coffee" / "clean" / "coffee_reviews"
    for day in (1, 2, 3):
        frame = pl.DataFrame({"v": [day]})
        write_table(frame, table, inputs={}, at=datetime(2026, 9, day, tzinfo=UTC))

    result = CliRunner().invoke(cli.app, ["prune", "--keep", "1"])

    assert result.exit_code == 0, result.output
    assert "clean/coffee_reviews: 2 partitions removed" in result.output


def test_prune_says_so_when_there_is_nothing_to_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COFFEE_DATA_DIR", str(tmp_path))
    (tmp_path / "coffee").mkdir(parents=True)

    result = CliRunner().invoke(cli.app, ["prune"])

    assert "nothing to prune" in result.output


def test_analysis_run_writes_studies_and_publishes_figures(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repo root is redirected on purpose: the command publishes figures into
    `docs/figures/`, so without this the suite overwrites the documentation with
    pictures of the 25-row fixture -- committed, and rendered in the README."""
    runner = CliRunner()
    runner.invoke(cli.app, ["data", "run"])
    runner.invoke(cli.app, ["ml", "features"])
    monkeypatch.setattr(
        "mlops_core.analysis.pipeline.load_champion",
        lambda *args, **kwargs: ServedModel(ConstantModel(), "1", "cache"),
    )
    monkeypatch.setattr(cli, "REPO_ROOT", data_dir / "checkout")
    (data_dir / "checkout" / "docs").mkdir(parents=True)

    result = runner.invoke(cli.app, ["analysis", "run"])

    assert result.exit_code == 0, result.output
    assert "feature_recommendation:" in result.output
    assert (data_dir / "coffee" / "analysis" / "feature_recommendation").is_dir()
    published = data_dir / "checkout" / "docs" / "figures"
    assert {path.name for path in published.iterdir()} == {
        "target_distribution.png",
        "feature_importance.png",
        "market_history.png",  # residual_bias needs predictions, which this run has none of
    }


def test_the_dashboard_command_launches_streamlit_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Started through us, not through a raw `streamlit run`, so the domain and the
    headless flag are always set: otherwise it blocks asking for an email."""
    launched: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "streamlit.web",
        types.SimpleNamespace(
            cli=types.SimpleNamespace(main=lambda: launched.update(argv=sys.argv))
        ),
    )

    CliRunner().invoke(cli.app, ["analysis", "dashboard", "--port", "9999"])

    assert "--server.headless" in launched["argv"]  # type: ignore[operator]
    assert "9999" in launched["argv"]  # type: ignore[operator]
    assert os.environ["COFFEE_DOMAIN"] == "coffee"


def test_secrets_reports_what_is_configured_without_printing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COFFEE_DENUE_TOKEN", "super-secret-token")
    monkeypatch.delenv("COFFEE_USDA_FAS_API_KEY", raising=False)

    result = CliRunner().invoke(cli.app, ["secrets"])

    assert result.exit_code == 0, result.output
    assert "set (18 characters)" in result.output
    assert "super-secret-token" not in result.output
    assert "USDA FAS key (COFFEE_USDA_FAS_API_KEY): missing" in result.output
