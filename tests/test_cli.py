import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import typer
from typer.testing import CliRunner

from coffee_mlops import cli
from coffee_mlops.data.extract import http_client
from coffee_mlops.ml.registry import ServedModel
from coffee_mlops.ml.train import TrainResult
from tests.fakes import RecordedServer


class ConstantModel:
    """A stand-in champion for the chained-run test."""

    def predict(self, x: object) -> list[float]:
        return [82.0] * len(x)  # type: ignore[arg-type]


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: RecordedServer) -> Path:
    """Point the CLI at a temp data dir and at the recorded server instead of the internet."""

    @contextmanager
    def recorded_client() -> Iterator[httpx.Client]:
        with http_client(httpx.MockTransport(server.handler)) as client:
            yield client

    monkeypatch.setattr(cli, "http_client", recorded_client)
    monkeypatch.setenv("COFFEE_DATA_DIR", str(tmp_path))
    return tmp_path


def test_extract_writes_raw_layer_under_the_domain(data_dir: Path) -> None:
    result = CliRunner().invoke(cli.app, ["data", "extract", "--domain", "coffee"])

    assert result.exit_code == 0, result.output
    raw = data_dir / "coffee" / "raw"
    assert {p.name for p in raw.iterdir()} == {"cqi_2018", "cqi_2023", "psd_coffee"}


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
    assert {p.name for p in clean.iterdir()} == {"coffee_reviews", "market_context"}


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
    }


def test_ml_run_chains_features_and_training(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    CliRunner().invoke(cli.app, ["data", "run"])
    monkeypatch.setattr(
        "coffee_mlops.ml.train.train_model",
        lambda config, data, uri: TrainResult("run-1", "1", False, {"test_mae": 2.0}),
    )
    monkeypatch.setattr(
        "coffee_mlops.ml.predict.load_champion",
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

    monkeypatch.setattr("coffee_mlops.ml.train.train_model", fake_train_model)
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
