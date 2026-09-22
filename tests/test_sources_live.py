"""Checks that the configured upstream URLs still answer. Hits the internet.

Excluded by default; run with `uv run pytest -m network`.
"""

from pathlib import Path

import pytest

import domains.coffee
from domains.coffee.sources.overpass import COUNT, fetch
from mlops_core.data.api import ApiClient
from mlops_core.data.extract import http_client

CONFIG = domains.coffee.adapter().config
SOURCES = CONFIG.sources


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(SOURCES))
def test_source_is_reachable(name: str) -> None:
    # GET, not HEAD: Kaggle answers HEAD with 404.
    with http_client() as client, client.stream("GET", str(SOURCES[name].url)) as response:
        assert response.status_code == 200
        assert next(response.iter_bytes())


@pytest.mark.network
def test_overpass_still_knows_the_area_and_the_tag(tmp_path: Path) -> None:
    """Asks for the count, not the inventory: this checks the selectors still match
    something, without pulling a thousand places off a volunteer-run service."""
    overpass = CONFIG.overpass
    assert overpass is not None
    with http_client() as client:
        api = ApiClient(client=client, cache_dir=tmp_path, min_interval_s=0.0)
        payload = fetch(api, overpass, COUNT)

    assert int(payload["elements"][0]["tags"]["total"]) > 0
