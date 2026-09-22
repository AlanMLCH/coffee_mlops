"""Coffee: quality of graded lots, the world market behind them, and where Mexico City
drinks it. The first domain, and the one the core was extracted from."""

from pathlib import Path

from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.config import CoffeeConfig
from mlops_core.config import load_config

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def adapter() -> CoffeeAdapter:
    """What `mlops_core.adapter.load_adapter("coffee")` calls."""
    return CoffeeAdapter(load_config(CONFIG_PATH, CoffeeConfig))
