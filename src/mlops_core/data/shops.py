"""Online shops' own catalogs, read the way the platform publishes them.

Many small shops run on a hosted platform that exposes the catalog as JSON: Shopify at
`/products.json`, Squarespace by adding `?format=json` to a store page. Reading that is
steadier than scraping the rendered pages - the theme can change every season, the JSON
shape does not - and it is what the platform serves to anyone. Every request is checked
against the site's robots.txt first.

Only what a platform's JSON leaves out is read from the rendered page, by the domain
that knows which pages it needs.
"""

import logging
from typing import Any

from mlops_core.data.api import ApiClient
from mlops_core.data.robots import RobotsPolicy

logger = logging.getLogger(__name__)

SHOPIFY_PAGE_SIZE = 250  # the most Shopify returns per page


def shopify_products(
    client: ApiClient, robots: RobotsPolicy, base_url: str
) -> list[dict[str, Any]]:
    """Every product of a Shopify store, page by page, stopping on the first short page."""
    products: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{base_url}/products.json?limit={SHOPIFY_PAGE_SIZE}&page={page}"
        robots.check(url)
        batch: list[dict[str, Any]] = client.get_json(url, cache_key=f"shopify:{url}")["products"]
        products += batch
        if len(batch) < SHOPIFY_PAGE_SIZE:
            return products
        page += 1


def squarespace_items(
    client: ApiClient, robots: RobotsPolicy, base_url: str, store_path: str
) -> list[dict[str, Any]]:
    """Every item of a Squarespace store page.

    A store small enough for one page answers with `pagination` empty. A larger one
    pages, and how it pages has not been verified against a real store yet: rather than
    guess the parameters and quietly stop after the first page, this refuses.
    """
    url = f"{base_url}{store_path}?format=json"
    robots.check(url)
    payload: dict[str, Any] = client.get_json(url, cache_key=f"squarespace:{url}")
    if payload.get("pagination"):
        raise NotImplementedError(
            f"{url} paginates; reading only its first page would truncate it silently"
        )
    items: list[dict[str, Any]] = payload.get("items", [])
    return items
