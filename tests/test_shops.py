"""Shop catalogs are read as the platform publishes them, to the end, and never guessed."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from mlops_core.data.api import ApiClient
from mlops_core.data.robots import RobotsDisallowed, RobotsPolicy
from mlops_core.data.shops import SHOPIFY_PAGE_SIZE, shopify_products, squarespace_items


class FakeShop:
    """A shop with `total` products, a robots.txt, and a record of what was asked."""

    def __init__(
        self, total: int = 0, robots: str = "User-agent: *\nDisallow:\n", store: Any = None
    ):
        self.total, self.robots, self.store = total, robots, store
        self.asked: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=self.robots)
        if request.url.path == "/tienda":
            return httpx.Response(200, json=self.store)
        page = int(request.url.params["page"])
        first = (page - 1) * SHOPIFY_PAGE_SIZE
        ids = range(first, min(first + SHOPIFY_PAGE_SIZE, self.total))
        return httpx.Response(200, json={"products": [{"id": i} for i in ids]})


def connect(shop: FakeShop, tmp_path: Path) -> tuple[ApiClient, RobotsPolicy]:
    client = ApiClient(
        client=httpx.Client(transport=httpx.MockTransport(shop.handler)),
        cache_dir=tmp_path,
        min_interval_s=0.0,
        sleep=lambda _: None,
    )
    return client, RobotsPolicy(client, "coffee-mlops/0.1.0")


def test_every_page_of_a_catalog_is_read_until_a_short_one(tmp_path: Path) -> None:
    shop = FakeShop(total=SHOPIFY_PAGE_SIZE + 30)

    products = shopify_products(*connect(shop, tmp_path), "https://shop.test")

    assert len(products) == SHOPIFY_PAGE_SIZE + 30
    assert shop.asked == ["/robots.txt", "/products.json", "/products.json"]


def test_a_catalog_robots_forbids_is_not_read(tmp_path: Path) -> None:
    shop = FakeShop(total=5, robots="User-agent: *\nDisallow: /products.json\n")

    with pytest.raises(RobotsDisallowed):
        shopify_products(*connect(shop, tmp_path), "https://shop.test")

    assert shop.asked == ["/robots.txt"]  # refused before a single product was asked for


def test_a_one_page_store_is_read_whole(tmp_path: Path) -> None:
    shop = FakeShop(store={"items": [{"id": "a"}, {"id": "b"}], "pagination": None})

    items = squarespace_items(*connect(shop, tmp_path), "https://shop.test", "/tienda")

    assert [item["id"] for item in items] == ["a", "b"]


def test_a_store_that_pages_is_refused_rather_than_truncated(tmp_path: Path) -> None:
    """How Squarespace pages has not been verified; reading page one and stopping would
    look exactly like a complete store."""
    shop = FakeShop(store={"items": [{"id": "a"}], "pagination": {"nextPage": True}})

    with pytest.raises(NotImplementedError, match="truncate"):
        squarespace_items(*connect(shop, tmp_path), "https://shop.test", "/tienda")
