"""A coffee shop, as the core sees it: `mlops_core.adapter.DomainAdapter`.

A subdomain of coffee, and a tenant of its own: its data, models, API and agent are the
shop's. What it knows of the coffee market - green coffee in pesos, the consumer price
index - it reads from its parent's tables, listed one by one in `parent`; of the other
shops it can read nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, SecretStr

from domains.coffee.shops.config import CoffeeShopConfig
from domains.coffee.shops.features import (
    HOURS_CONTEXT,
    ORDERS_CONTEXT,
    add_hour_context,
    add_order_context,
)
from domains.coffee.shops.request import ShopHour, ShopOrder
from domains.coffee.shops.schemas import CLEAN_SCHEMAS, RAW_SCHEMAS
from mlops_core.adapter import ApiExtraction, CleanTable, FileReader, JsonReader

if TYPE_CHECKING:
    import httpx
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class ModelHooks:
    """What one of the shop's models needs from code: its context, its enrichment and
    the API's request body."""

    context_tables: tuple[str, ...]
    enrich: Callable[[pl.DataFrame, Mapping[str, pl.DataFrame]], pl.DataFrame]
    request: type[BaseModel]


MODELS = {
    # How many tickets an hour brings: its day, its hour and the menu's prices are its context.
    "hourly_demand": ModelHooks(
        context_tables=HOURS_CONTEXT,
        enrich=add_hour_context,
        request=ShopHour,
    ),
    # How long an order takes from order to ready: its hour's crowd is its context.
    "order_minutes": ModelHooks(
        context_tables=ORDERS_CONTEXT,
        enrich=add_order_context,
        request=ShopOrder,
    ),
}


class CoffeeShopAdapter:
    """One coffee shop in Mexico City, simulated from real anchors."""

    def __init__(self, config: CoffeeShopConfig):
        self._config = config

    @property
    def config(self) -> CoffeeShopConfig:
        return self._config

    def credentials(self) -> Mapping[str, SecretStr | None]:
        return {}  # every source is open, and the export is the shop's own

    def extract(
        self, data_dir: Path, client: httpx.Client, now: datetime | None = None
    ) -> ApiExtraction:
        """The point-of-sale export. A real shop's would be read from its system here;
        a demo shop's is simulated from real anchors and its parent's tables."""
        from domains.coffee.shops.simulate import extract
        from mlops_core.storage import latest_partition, read_table, table_path

        config = self.config
        lent = {
            name: read_table(table_path(data_dir, config.readable(name)))
            for name in (config.simulation.green_coffee, config.simulation.inflation)
            if latest_partition(table_path(data_dir, name)) is not None
        }
        return extract(config.shop, config.simulation, data_dir, lent, now)

    def raw_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        return RAW_SCHEMAS

    def json_readers(self) -> Mapping[str, JsonReader]:
        from domains.coffee.shops.simulate import TABLES, to_frame

        return dict.fromkeys(TABLES, to_frame)

    def file_readers(self) -> Mapping[str, FileReader]:
        return {}

    def clean(
        self, raw: Mapping[str, pl.DataFrame], read_at: Mapping[str, datetime]
    ) -> Mapping[str, CleanTable]:
        from domains.coffee.shops.clean import clean_shop
        from domains.coffee.shops.simulate import TABLES

        missing = [table for table in TABLES if table not in raw]
        if missing:  # an empty layer would look like a shop that sold nothing
            raise ValueError(
                f"No point-of-sale export ({missing}): build the coffee domain's tables it "
                "reads, then run extract for the shop"
            )
        export = tuple(sorted(TABLES))
        tables = clean_shop(dict(raw), self.config.shop)
        return {name: CleanTable(frame, export) for name, frame in tables.items()}

    def clean_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        return CLEAN_SCHEMAS

    def context_tables(self, model: str) -> tuple[str, ...]:
        return hooks(model).context_tables

    def enrich(
        self, model: str, items: pl.DataFrame, context: Mapping[str, pl.DataFrame]
    ) -> pl.DataFrame:
        return hooks(model).enrich(items, context)

    def request_model(self, model: str) -> type[BaseModel]:
        return hooks(model).request

    def studies(self, clean: Mapping[str, pl.DataFrame]) -> Mapping[str, pl.DataFrame]:
        from domains.coffee.shops.analysis import studies

        return studies(clean, self.config)

    def figures(self, tables: Mapping[str, pl.DataFrame]) -> Mapping[str, Figure]:
        return {}


def hooks(model: str) -> ModelHooks:
    """The code behind one of the YAML's models; a model with none is a config error."""
    if model not in MODELS:
        raise ValueError(f"The coffee shop has no code for model '{model}'; it has {list(MODELS)}")
    return MODELS[model]
