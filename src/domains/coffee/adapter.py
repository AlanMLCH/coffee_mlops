"""The coffee domain, as the core sees it: `mlops_core.adapter.DomainAdapter`.

Everything the pipeline cannot know on its own about coffee is answered here, mostly by
pointing at the module that already knows it. What needs the data pipeline's heavier
dependencies (httpx, DuckDB's spatial extension, matplotlib) is imported inside the
method that uses it: the prediction API loads this adapter too, and its image installs
none of them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, SecretStr

from domains.coffee.config import CoffeeConfig, CoffeeCredentials
from domains.coffee.features import (
    CONTEXT_TABLE,
    ORIGINS_TABLE,
    add_coffee_origin,
    add_market_context,
)
from domains.coffee.forecast import PRICES_TABLE, add_price_history
from domains.coffee.request import Lot, Offer, PriceMonth
from domains.coffee.schemas import (
    RAW_SCHEMAS,
    borough_profile_schema,
    clean_schemas,
    intercensal_schema,
)
from mlops_core.adapter import ApiExtraction, CleanTable, FileReader, JsonReader

if TYPE_CHECKING:
    import httpx
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class ModelHooks:
    """What one of the domain's models needs from code: its context, its enrichment and
    the API's request body. The rest of a model is data, in the YAML."""

    context_tables: tuple[str, ...]
    enrich: Callable[[pl.DataFrame, Mapping[str, pl.DataFrame]], pl.DataFrame]
    request: type[BaseModel]


MODELS = {
    # A graded lot sees its origin country's market balance of the year before grading.
    "review": ModelHooks(
        context_tables=(CONTEXT_TABLE,),
        enrich=lambda items, context: add_market_context(items, context[CONTEXT_TABLE]),
        request=Lot,
    ),
    # A bag on a shop's shelf is described by its coffee's sheet: where it grew, how.
    "offer": ModelHooks(
        context_tables=(ORIGINS_TABLE,),
        enrich=lambda items, context: add_coffee_origin(items, context[ORIGINS_TABLE]),
        request=Offer,
    ),
    # A month's green coffee price, from the months before it: the history is its context.
    "green_price": ModelHooks(
        context_tables=(PRICES_TABLE,),
        enrich=lambda items, context: add_price_history(items, context[PRICES_TABLE]),
        request=PriceMonth,
    ),
}


class CoffeeAdapter:
    """Graded coffee lots, the world market behind them, and Mexico City's coffee shops."""

    def __init__(self, config: CoffeeConfig, credentials: CoffeeCredentials | None = None):
        self._config = config
        # Read when needed, not at import: the environment of a test or a container can
        # differ from the one this module was first imported in.
        self._credentials = credentials

    @property
    def config(self) -> CoffeeConfig:
        return self._config

    @property
    def keys(self) -> CoffeeCredentials:
        return self._credentials or CoffeeCredentials()

    def credentials(self) -> Mapping[str, SecretStr | None]:
        keys = self.keys
        return {
            "DENUE token (COFFEE_DENUE_TOKEN)": keys.denue_token,
            "USDA FAS key (COFFEE_USDA_FAS_API_KEY)": keys.usda_fas_api_key,
        }

    def extract(
        self, data_dir: Path, client: httpx.Client, now: datetime | None = None
    ) -> ApiExtraction:
        from domains.coffee.sources import extract

        return extract(self.config, self.keys, data_dir, client, now)

    def raw_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        # A closed year of the shelf survey is the same file as the year in course; the
        # survey of who lives where is read for the columns its profile names.
        closed = self.config.consumer_prices.closed_years
        profile = self.config.borough_profile
        return {
            **RAW_SCHEMAS,
            **dict.fromkeys(closed, RAW_SCHEMAS["profeco_prices"]),
            profile.source: intercensal_schema(profile),
        }

    def json_readers(self) -> Mapping[str, JsonReader]:
        from domains.coffee.sources import denue, fas, overpass, roasters

        config = self.config
        readers: dict[str, JsonReader] = {}
        if config.denue is not None:
            readers[config.denue.name] = denue.to_frame
            if config.denue.workplaces is not None:
                readers[config.denue.workplaces.name] = denue.workplaces_frame
        if config.overpass is not None:
            readers[config.overpass.name] = overpass.to_frame
        if config.fas is not None:
            readers[config.fas.name] = fas.to_frame
        if config.roasters is not None:
            readers[config.roasters.name] = roasters.to_frame
        return readers

    def file_readers(self) -> Mapping[str, FileReader]:
        from domains.coffee.sources.faostat import read_producer_prices
        from domains.coffee.sources.ico import read_indicator_prices
        from domains.coffee.sources.profeco import read_shelf_prices

        shelves, farmers = self.config.consumer_prices, self.config.producer_prices
        readers: dict[str, FileReader] = {"ico_prices": read_indicator_prices}
        # CoffeeConfig refuses a producer price source without a member.
        readers[farmers.source] = partial(read_producer_prices,
                                          member=str(self.config.sources[farmers.source].member),
                                          item_code=farmers.item_code)  # fmt: skip
        for name in (shelves.source, *shelves.closed_years):
            folder = self.config.sources[name].member
            if folder is None:  # pragma: no cover - CoffeeConfig refuses such a config
                raise ValueError(f"{name} names no folder of fortnights")
            readers[name] = partial(read_shelf_prices, folder=folder,
                                    products=list(shelves.products))  # fmt: skip
        return readers

    def clean(
        self, raw: Mapping[str, pl.DataFrame], read_at: Mapping[str, datetime]
    ) -> Mapping[str, CleanTable]:
        from domains.coffee.clean import clean_tables

        config = self.config
        return clean_tables(
            raw,
            config.cleaning,
            config.production,
            config.consumer_prices,
            config.producer_prices,
            config.borough_profile,
            read_at,
        )

    def clean_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        profile = borough_profile_schema(self.config.borough_profile)
        return {**clean_schemas(self.config.cleaning), "borough_profile": profile}

    def context_tables(self, model: str) -> tuple[str, ...]:
        return hooks(model).context_tables

    def enrich(
        self, model: str, items: pl.DataFrame, context: Mapping[str, pl.DataFrame]
    ) -> pl.DataFrame:
        return hooks(model).enrich(items, context)

    def request_model(self, model: str) -> type[BaseModel]:
        return hooks(model).request

    def studies(self, clean: Mapping[str, pl.DataFrame]) -> Mapping[str, pl.DataFrame]:
        from domains.coffee.analysis import studies

        config = self.config
        return studies(
            clean,
            config.market_analysis,
            config.production,
            config.consumer_prices,
            config.cleaning.roaster_sheets.states,
            config.cleaning.roaster_sheets.home_country,
            config.analysis.min_rows,
            config.cleaning.register_match.radius_m,
        )

    def figures(self, tables: Mapping[str, pl.DataFrame]) -> Mapping[str, Figure]:
        from domains.coffee.analysis import figures

        return figures(tables, self.config.market_analysis)


def hooks(model: str) -> ModelHooks:
    """The code behind one of the YAML's models; a model with none is a config error."""
    if model not in MODELS:
        raise ValueError(f"Coffee has no code for model '{model}'; it has {list(MODELS)}")
    return MODELS[model]
