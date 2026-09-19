"""Checks that the configured upstream URLs still answer. Hits the internet.

Excluded by default; run with `uv run pytest -m network`.
"""

import pytest

from coffee_mlops.config import load_domain_config
from coffee_mlops.extract import http_client

SOURCES = load_domain_config("coffee").sources


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(SOURCES))
def test_source_is_reachable(name: str) -> None:
    # GET, not HEAD: Kaggle answers HEAD with 404.
    with http_client() as client, client.stream("GET", str(SOURCES[name].url)) as response:
        assert response.status_code == 200
        assert next(response.iter_bytes())
