import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from coffee_mlops import cli
from coffee_mlops.extract import http_client
from tests.fakes import RecordedServer


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
    result = CliRunner().invoke(cli.app, ["extract", "--domain", "coffee"])

    assert result.exit_code == 0, result.output
    raw = data_dir / "coffee" / "raw"
    assert {p.name for p in raw.iterdir()} == {"cqi_2018", "cqi_2023", "psd_coffee"}


def test_validate_runs_after_extract(data_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli.app, ["extract"])

    result = runner.invoke(cli.app, ["validate", "--domain", "coffee"])

    assert result.exit_code == 0, result.output
    assert "psd_coffee: 114 rows valid" in result.output


def test_clean_builds_the_clean_layer(data_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli.app, ["extract"])

    result = runner.invoke(cli.app, ["clean", "--domain", "coffee"])

    assert result.exit_code == 0, result.output
    clean = data_dir / "coffee" / "clean"
    assert {p.name for p in clean.iterdir()} == {"coffee_reviews", "market_context"}


def test_features_and_sql_run_on_the_built_layers(data_dir: Path) -> None:
    runner = CliRunner()
    for step in ("extract", "clean", "features"):
        assert runner.invoke(cli.app, [step]).exit_code == 0

    result = runner.invoke(cli.app, ["sql", "SELECT count(*) AS n FROM features.review_features"])

    assert result.exit_code == 0, result.output
    assert "25" in result.output


def test_request_urls_are_not_logged(data_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Signed URLs (and, from stage 2, API tokens) travel in URLs; they must not hit logs."""
    caplog.set_level(logging.DEBUG)

    CliRunner().invoke(cli.app, ["extract"])

    assert "Signature" not in caplog.text
