"""Coffee: quality of graded lots, the world market behind them, and where Mexico City
drinks it - and, as its subdomains, the coffee shops that use all of it. The first domain,
and the one the core was extracted from."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.config import CoffeeConfig
from mlops_core.config import load_config

if TYPE_CHECKING:
    from domains.coffee.shops.adapter import CoffeeShopAdapter

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def adapter() -> CoffeeAdapter:
    """What `mlops_core.adapter.load_adapter("coffee")` calls."""
    return CoffeeAdapter(load_config(CONFIG_PATH, CoffeeConfig))


# Its subdomains are coffee shops, one per business: each a tenant that reads the coffee
# tables it lists and nothing of the others (`domains/coffee/shops`). Imported when asked
# for: the coffee domain itself never needs them.


def subdomains() -> list[str]:
    """Every coffee shop, by name."""
    from domains.coffee import shops

    return shops.businesses()


def subdomain(name: str) -> CoffeeShopAdapter:
    """What `load_adapter("coffee/<shop>")` calls."""
    from domains.coffee import shops

    return shops.adapter(name)


def subdomain_dir(name: str) -> Path:
    """Where a shop's files live: every shop's, beside their code."""
    from domains.coffee import shops

    return shops.SHOPS_DIR
