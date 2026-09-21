"""One module per API source, each concrete about the service it talks to.

They share the polite client in `data/api.py`; the contract they have in common is
extracted at the end of stage 2, once three of them exist.

`extract_api_sources` is the one place that knows which sources exist and what each
needs. It lives here, not in the CLI, for the reason the DENUE token leak taught: a
step that only runs when you came through one entry point is a step the orchestrator,
a notebook or a script quietly skips.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from mlops_core.config import DomainConfig, Settings
from mlops_core.data.api import ApiClient
from mlops_core.data.extract import RawArtifact
from mlops_core.data.sources.denue import ingest_establishments
from mlops_core.data.sources.fas import ingest_balance
from mlops_core.data.sources.overpass import ingest_places

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiExtraction:
    """What came back, and what did not: a missing credential is reported, not raised.

    A fresh clone has no `.env`, and the whole of stage 1 must still build. Skipping
    has to be loud, though, or an empty layer looks like an upstream with no rows.
    """

    artifacts: dict[str, RawArtifact] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # source name -> why


def extract_api_sources(
    config: DomainConfig,
    settings: Settings,
    data_dir: Path,
    client: httpx.Client,
    now: datetime | None = None,
) -> ApiExtraction:
    """Pull every API source the domain declares, into `<data_dir>/raw`."""
    result = ApiExtraction()
    raw_dir = data_dir / "raw"

    if (denue := config.denue) is not None:
        if settings.denue_token is None:
            result.skipped[denue.name] = "COFFEE_DENUE_TOKEN is not set"
        else:
            api = _client(client, data_dir, denue.name, denue.rate_limit_seconds, denue.cache_hours)
            token = settings.denue_token.get_secret_value()
            result.artifacts[denue.name] = ingest_establishments(api, denue, token, raw_dir, now)

    if (overpass := config.overpass) is not None:
        api = _client(
            client, data_dir, overpass.name, overpass.rate_limit_seconds, overpass.cache_hours
        )
        result.artifacts[overpass.name] = ingest_places(api, overpass, raw_dir, now)

    if (fas := config.fas) is not None:
        if settings.usda_fas_api_key is None:
            result.skipped[fas.name] = "COFFEE_USDA_FAS_API_KEY is not set"
        else:
            api = _client(client, data_dir, fas.name, fas.rate_limit_seconds, fas.cache_hours)
            key = settings.usda_fas_api_key.get_secret_value()
            result.artifacts[fas.name] = ingest_balance(api, fas, key, raw_dir, now)

    for name, reason in result.skipped.items():
        logger.warning("%s skipped: %s", name, reason)
    return result


def _client(
    client: httpx.Client, data_dir: Path, name: str, interval_s: float, cache_hours: float
) -> ApiClient:
    """One cache directory per source, so one service's answers never shadow another's."""
    return ApiClient(
        client=client,
        cache_dir=data_dir / "cache" / name,
        min_interval_s=interval_s,
        max_age_s=cache_hours * 3600,
    )
