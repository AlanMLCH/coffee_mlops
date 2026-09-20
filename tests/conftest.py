"""Shared fixtures. HTTP is replayed from recorded payloads in tests/fixtures/."""

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from coffee_mlops.config import DomainConfig, load_domain_config
from coffee_mlops.data.extract import extract_all, http_client
from tests.fakes import RecordedServer

FIXTURES = Path(__file__).parent / "fixtures"

# Kaggle answers with a 302 to a short-lived signed URL like this one.
SIGNED_URL = "https://storage.googleapis.com/kaggle-data-sets/archive.zip?X-Goog-Signature=abc123"


def zip_fixture(fixture: str, member: str) -> bytes:
    """Rebuild the upstream ZIP envelope around a recorded CSV excerpt."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, (FIXTURES / fixture).read_bytes())
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def never_reach_a_real_mlflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A developer running a local MLflow must not change what the tests exercise:
    point every test at an empty throwaway registry unless it sets its own."""
    monkeypatch.setenv(
        "COFFEE_MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'unused-mlflow.db').as_posix()}"
    )


@pytest.fixture
def coffee_config() -> DomainConfig:
    return load_domain_config("coffee")


@pytest.fixture
def recorded() -> dict[str, bytes]:
    """Recorded response body per source name."""
    return {
        "cqi_2018": (FIXTURES / "cqi_2018_sample.csv").read_bytes(),
        "cqi_2023": zip_fixture("cqi_2023_sample.csv", "df_arabica_clean.csv"),
        "psd_coffee": zip_fixture("psd_coffee_sample.csv", "psd_coffee.csv"),
    }


@pytest.fixture
def server(coffee_config: DomainConfig, recorded: dict[str, bytes]) -> RecordedServer:
    urls = {name: str(source.url) for name, source in coffee_config.sources.items()}
    payloads = {urls[name]: body for name, body in recorded.items() if name != "cqi_2023"}
    payloads[SIGNED_URL] = recorded["cqi_2023"]
    return RecordedServer(payloads, redirects={urls["cqi_2023"]: SIGNED_URL})


@pytest.fixture
def client(server: RecordedServer) -> Iterator[httpx.Client]:
    with http_client(httpx.MockTransport(server.handler)) as c:
        yield c


@pytest.fixture
def raw_dir(tmp_path: Path, coffee_config: DomainConfig, client: httpx.Client) -> Path:
    """A raw layer populated from the recorded payloads, in the real directory layout
    (<data_dir>/<domain>/raw), so `raw_dir.parent` is the domain's data dir."""
    raw = tmp_path / coffee_config.name / "raw"
    extract_all(coffee_config, raw, client)
    return raw
