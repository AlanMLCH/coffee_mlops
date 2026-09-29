"""Checks that the configured upstream URLs still answer. Hits the internet.

Excluded by default; run with `uv run pytest -m network`.
"""

from pathlib import Path

import pytest

import domains.coffee
from domains.coffee.config import ShopConfig
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


# The documents a publisher serves to anyone: their addresses move (a report is replaced,
# a catalogue republished). The ones handed over by hand are not fetched at all.
DOCUMENTS = {document.name: document for document in CONFIG.documents if document.inbox is None}


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(DOCUMENTS))
def test_document_is_reachable(name: str) -> None:
    with http_client() as client, client.stream("GET", str(DOCUMENTS[name].url)) as response:
        assert response.status_code == 200
        assert next(response.iter_bytes())


@pytest.mark.network
@pytest.mark.parametrize(
    "shop", CONFIG.roasters.shops if CONFIG.roasters else [], ids=lambda shop: shop.shop
)
def test_a_roasters_catalogue_still_has_the_shape_it_is_read_in(shop: ShopConfig) -> None:
    """One request a shop, a week: a platform that moved its catalogue, or renamed the key
    the listings sit under, would otherwise surface as an empty catalogue."""
    if shop.platform == "shopify":
        url, key = f"{shop.base_url}/products.json?limit=1", "products"
    else:
        url, key = f"{shop.base_url}{shop.store_path}?format=json", "items"
    with http_client() as client:
        response = client.get(url)

    assert response.status_code == 200
    assert response.json()[key]
