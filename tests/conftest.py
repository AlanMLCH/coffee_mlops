"""Shared fixtures. HTTP is replayed from recorded payloads in tests/fixtures/.

`httpx` and the ETL package are imported inside the fixtures that need them, so the
serving tests can run in an environment that installed only the `serving` extra. That
is how CI proves the API's dependency list is complete instead of relying on packages
another pipeline happens to pull in.
"""

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

import domains.coffee
from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.config import CoffeeConfig, CoffeeCredentials
from mlops_core.config import Settings
from tests.fakes import RecordedServer, fas_recording, without_rate_limits

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
def isolate_from_the_developers_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """What is installed or configured on this machine must not change what the tests
    exercise: no local MLflow, and no reading the developer's `.env`, which holds real
    credentials and would make a test about a missing one pass or fail by accident."""
    monkeypatch.setenv(
        "MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'unused-mlflow.db').as_posix()}"
    )
    for settings in (Settings, CoffeeCredentials):
        monkeypatch.setitem(settings.model_config, "env_file", None)
    # Which domain runs must come from the test, never from the machine it runs on.
    monkeypatch.delenv("MLOPS_DOMAIN", raising=False)


@pytest.fixture
def coffee_adapter() -> CoffeeAdapter:
    """The real coffee domain, as the core loads it."""
    return domains.coffee.adapter()


@pytest.fixture
def coffee_config(coffee_adapter: CoffeeAdapter) -> CoffeeConfig:
    return coffee_adapter.config


# Every credential the domain can use, so fixtures exercise every source.
FIXTURE_CREDENTIALS = CoffeeCredentials(
    denue_token=SecretStr("fixture-token"), usda_fas_api_key=SecretStr("fixture-key")
)


@pytest.fixture
def recorded() -> dict[str, bytes]:
    """Recorded response body per source name."""
    return {
        "cqi_2018": (FIXTURES / "cqi_2018_sample.csv").read_bytes(),
        "cqi_2023": zip_fixture("cqi_2023_sample.csv", "df_arabica_clean.csv"),
        "psd_coffee": zip_fixture("psd_coffee_sample.csv", "psd_coffee.csv"),
        # Shaped like INEGI's download: a shapefile inside a ZIP, Latin-1 attributes,
        # the layer's own projection. Two boroughs instead of sixteen.
        "cdmx_boroughs": (FIXTURES / "cdmx_boroughs_sample.zip").read_bytes(),
        # SIAP's real bytes, still Latin-1: 11 coffee rows (Ocosingo's three CADERs among
        # them) and two other crops, one with nothing harvested.
        "siap_agricola": (FIXTURES / "siap_agricola_sample.csv").read_bytes(),
    }


@pytest.fixture
def server(coffee_config: CoffeeConfig, recorded: dict[str, bytes]) -> RecordedServer:
    urls = {name: str(source.url) for name, source in coffee_config.sources.items()}
    payloads = {urls[name]: body for name, body in recorded.items() if name != "cqi_2023"}
    payloads[SIGNED_URL] = recorded["cqi_2023"]
    return RecordedServer(
        payloads,
        redirects={urls["cqi_2023"]: SIGNED_URL},
        overpass=(FIXTURES / "overpass_places_sample.json").read_bytes(),
        fas=fas_recording(),
    )


@pytest.fixture
def client(server: RecordedServer) -> Iterator[Any]:
    import httpx

    from mlops_core.data.extract import http_client

    with http_client(httpx.MockTransport(server.handler)) as c:
        yield c


@pytest.fixture
def raw_dir(tmp_path: Path, coffee_config: CoffeeConfig, client: Any) -> Path:
    """A raw layer populated from the recorded payloads, in the real directory layout
    (<data_dir>/<domain>/raw), so `raw_dir.parent` is the domain's data dir.

    The API sources are pulled too, with a token supplied: the clean layer's table of
    places is built from them, so a fixture without them would test half a pipeline.
    """
    from mlops_core.data.extract import extract_all

    data_dir = tmp_path / coffee_config.name
    extract_all(coffee_config, data_dir / "raw", client)
    adapter = CoffeeAdapter(without_rate_limits(coffee_config), FIXTURE_CREDENTIALS)
    adapter.extract(data_dir, client)
    return data_dir / "raw"
