"""One module per API source, each concrete about the service it talks to.

They share the core's polite client (`mlops_core.data.api`). `extract` is the one place
that knows which of them exist and what each needs; the adapter hands it to the core,
so the CLI and the orchestrator run exactly the same extraction. It must not live in an
entry point, for the reason the DENUE token leak taught: a step that only runs when you
came through one door is a step every other door quietly skips.
"""

import logging
from datetime import datetime
from pathlib import Path

import httpx

from domains.coffee.config import CoffeeConfig, CoffeeCredentials
from domains.coffee.sources.denue import ingest_establishments
from domains.coffee.sources.fas import ingest_balance
from domains.coffee.sources.overpass import ingest_places
from mlops_core.adapter import ApiExtraction
from mlops_core.data.api import ApiClient

logger = logging.getLogger(__name__)


def extract(
    config: CoffeeConfig,
    credentials: CoffeeCredentials,
    data_dir: Path,
    client: httpx.Client,
    now: datetime | None = None,
) -> ApiExtraction:
    """Pull every API source the domain declares, into `<data_dir>/raw`."""
    result = ApiExtraction()
    raw_dir = data_dir / "raw"

    if (denue := config.denue) is not None:
        if credentials.denue_token is None:
            result.skipped[denue.name] = "COFFEE_DENUE_TOKEN is not set"
        else:
            api = _client(client, data_dir, denue.name, denue.rate_limit_seconds, denue.cache_hours)
            token = credentials.denue_token.get_secret_value()
            result.artifacts[denue.name] = ingest_establishments(api, denue, token, raw_dir, now)

    if (overpass := config.overpass) is not None:
        api = _client(
            client, data_dir, overpass.name, overpass.rate_limit_seconds, overpass.cache_hours
        )
        result.artifacts[overpass.name] = ingest_places(api, overpass, raw_dir, now)

    if (fas := config.fas) is not None:
        if credentials.usda_fas_api_key is None:
            result.skipped[fas.name] = "COFFEE_USDA_FAS_API_KEY is not set"
        else:
            api = _client(client, data_dir, fas.name, fas.rate_limit_seconds, fas.cache_hours)
            key = credentials.usda_fas_api_key.get_secret_value()
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
