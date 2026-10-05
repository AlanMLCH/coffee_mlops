"""The coffee domain's shops: a subdomain per business, each a tenant of its own - its
sales, prices, costs and hours - that reads the coffee tables listed in `parent` and
nothing of the other shops.

A shop is its file under `businesses/`: its menu, hours, staff, corner and history. What
every shop shares - the sources, the simulation, the models - is `config.yaml`. The Nth
shop costs a file, not code.
"""

from pathlib import Path
from typing import Any

import yaml

from domains.coffee.shops.adapter import CoffeeShopAdapter
from domains.coffee.shops.config import CoffeeShopConfig

SHOPS_DIR = Path(__file__).parent
BUSINESSES = SHOPS_DIR / "businesses"
CONFIG_PATH = SHOPS_DIR / "config.yaml"


def businesses() -> list[str]:
    """Every shop, by the name of its file."""
    return sorted(path.stem for path in BUSINESSES.glob("*.yaml"))


def shop_config(name: str) -> CoffeeShopConfig:
    """The shop called `name`: what every shop shares, with its own file's `shop` over
    it and its models registered under its name (`<name>-<model>`), so two shops never
    share a champion."""
    shared: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    own: dict[str, Any] = yaml.safe_load((BUSINESSES / f"{name}.yaml").read_text(encoding="utf-8"))
    if set(own) != {"shop"}:
        keys = sorted(own)
        raise ValueError(f"businesses/{name}.yaml describes its shop and nothing else: {keys}")
    prefix = name.replace("_", "-")
    for model in shared["models"]:
        training = model["training"]
        training["registered_model"] = f"{prefix}-{training['registered_model']}"
    # Its explorer is titled after it and opens on it.
    shop = own["shop"]
    shared["explore"]["title"] = shop["name"]
    shared["explore"]["view"] |= {"latitude": shop["latitude"], "longitude": shop["longitude"]}
    return CoffeeShopConfig.model_validate({**shared, **own, "name": name})


def adapter(name: str) -> CoffeeShopAdapter:
    """The shop called `name`, as the core sees it."""
    return CoffeeShopAdapter(shop_config(name))
