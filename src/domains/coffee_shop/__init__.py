"""The coffee shop: one neighbourhood specialty shop's sales, prices, costs and hours -
a tenant of its own that reads the coffee market from the coffee domain's tables."""

from pathlib import Path

from domains.coffee_shop.adapter import CoffeeShopAdapter
from domains.coffee_shop.config import CoffeeShopConfig
from mlops_core.config import load_config

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def adapter() -> CoffeeShopAdapter:
    """What `mlops_core.adapter.load_adapter("coffee_shop")` calls."""
    return CoffeeShopAdapter(load_config(CONFIG_PATH, CoffeeShopConfig))
