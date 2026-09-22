"""Mexico City's specialty roasters: what they sell, as their own shops list it.

This is the source that makes the domain current. The CQI data stops in 2023 and is
graded lots from anywhere; a roaster's bag is 2026, mostly Mexican, and names the same
things a CQI review does - origin, variety, process, altitude - plus a price.

Each shop is read the way its platform publishes the catalog (`mlops_core.data.shops`),
after its robots.txt allows it. A shop whose catalog JSON leaves the attributes out
(Buna keeps them only on the product page) also has its coffee product pages read. What
is stored is what the shops answered - the products as their platform returned them and
the pages as served - so the parsing can change without asking the shops again.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from domains.coffee.config import RoastersConfig, ShopConfig
from mlops_core.data.api import ApiClient
from mlops_core.data.extract import RawArtifact, store_payload
from mlops_core.data.robots import RobotsDisallowed, RobotsPolicy
from mlops_core.data.shops import shopify_products, squarespace_items

logger = logging.getLogger(__name__)


def is_coffee(product: dict[str, Any], shop: ShopConfig) -> bool:
    """Whether a listing is coffee, by the shop's own labels.

    Shops sell cups, t-shirts and subscriptions beside their coffee, and each labels
    things its own way: a product type, a tag, or nothing at all (a shop that sells only
    coffee). The config says which, per shop; titles it lists are excluded first.
    """
    title = product.get("title") or ""
    if any(re.search(pattern, title, re.IGNORECASE) for pattern in shop.exclude_titles):
        return False
    if not shop.product_types and not shop.tags:
        return True
    tags = set(product.get("tags") or [])
    return product.get("product_type") in shop.product_types or bool(tags & set(shop.tags))


def product_url(product: dict[str, Any], shop: ShopConfig) -> str:
    if shop.platform == "shopify":
        return f"{shop.base_url}/products/{product['handle']}"
    return f"{shop.base_url}{product['fullUrl']}"


def read_shop(client: ApiClient, robots: RobotsPolicy, shop: ShopConfig) -> dict[str, Any]:
    """One shop's coffee listings, and the product pages the config says to read."""
    if shop.platform == "shopify":
        listed = shopify_products(client, robots, shop.base_url)
    else:
        assert shop.store_path is not None  # the config model requires it for Squarespace
        listed = squarespace_items(client, robots, shop.base_url, shop.store_path)
    products = sorted((p for p in listed if is_coffee(p, shop)), key=lambda p: str(p["id"]))
    urls = {str(p["id"]): product_url(p, shop) for p in products}
    pages: dict[str, str] = {}
    if shop.product_pages:
        for product_id, url in urls.items():
            robots.check(url)
            pages[product_id] = client.get_text(url, cache_key=f"page:{url}")
    logger.info(
        "%s: %d coffee listings of %d, %d pages read",
        shop.shop,
        len(products),
        len(listed),
        len(pages),
    )
    return {"platform": shop.platform, "products": products, "urls": urls, "pages": pages}


def ingest_catalogs(
    clients: dict[str, ApiClient],
    robots: dict[str, RobotsPolicy],
    config: RoastersConfig,
    raw_dir: Path,
    now: datetime | None = None,
) -> tuple[RawArtifact, dict[str, str]]:
    """Every configured shop's coffee in one raw document, and the shops skipped and why.

    A shop whose robots.txt says no is skipped out loud, never read around: the others
    are still stored, and the skip is part of the record.
    """
    shops: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for shop in config.shops:
        try:
            shops[shop.shop] = read_shop(clients[shop.shop], robots[shop.shop], shop)
        except RobotsDisallowed as refused:
            skipped[shop.shop] = str(refused)
            logger.warning("%s skipped: %s", shop.shop, refused)
    document = {"shops": shops, "skipped": skipped}
    body = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
    # No single endpoint: the manifest names the shops the document was read from.
    origins = ", ".join(shop.base_url for shop in config.shops)
    artifact = store_payload(config.name, config.filename, body, raw_dir, origins, now)
    return artifact, skipped


def to_frame(document: dict[str, Any]) -> pl.DataFrame:
    """One row per offer - a product in one size - reshaped and not edited.

    The two platforms name the same things differently: Shopify's variant has a `price`
    and `grams`, Squarespace's a `priceMoney` and no weight this project can trust (its
    unit is not verified). Prices stay the text the shop sent; the contract types them.
    """
    rows = []
    for shop, catalog in document["shops"].items():
        shopify = catalog["platform"] == "shopify"
        for product in catalog["products"]:
            product_id = str(product["id"])
            for variant in product.get("variants") or []:
                rows.append(
                    {
                        "shop": shop,
                        "platform": catalog["platform"],
                        "product_id": product_id,
                        "variant_id": str(variant["id"]),
                        "title": product.get("title"),
                        "variant_title": variant.get("title") if shopify else _options(variant),
                        "price": variant.get("price")
                        if shopify
                        else variant["priceMoney"]["value"],
                        "platform_grams": variant.get("grams") if shopify else None,
                        "url": catalog["urls"][product_id],
                        "tags": ", ".join(product.get("tags") or []) or None,
                        "body_html": product.get("body_html") if shopify else product.get("body"),
                        "page_html": catalog["pages"].get(product_id),
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def _options(variant: dict[str, Any]) -> str | None:
    """Squarespace describes a variant by its attributes, e.g. {"Molienda": "En grano"}."""
    attributes = variant.get("attributes") or {}
    return " / ".join(str(value) for value in attributes.values()) or None
