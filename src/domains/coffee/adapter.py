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
from pathlib import Path
from typing import TYPE_CHECKING

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, SecretStr

from domains.coffee.config import CoffeeConfig, CoffeeCredentials
from domains.coffee.features import CONTEXT_TABLE, add_market_context
from domains.coffee.request import Lot
from domains.coffee.schemas import RAW_SCHEMAS, clean_schemas
from mlops_core.adapter import ApiExtraction, CleanTable, JsonReader

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
        return RAW_SCHEMAS

    def json_readers(self) -> Mapping[str, JsonReader]:
        from domains.coffee.sources import denue, fas, overpass, roasters

        config = self.config
        readers: dict[str, JsonReader] = {}
        if config.denue is not None:
            readers[config.denue.name] = denue.to_frame
        if config.overpass is not None:
            readers[config.overpass.name] = overpass.to_frame
        if config.fas is not None:
            readers[config.fas.name] = fas.to_frame
        if config.roasters is not None:
            readers[config.roasters.name] = roasters.to_frame
        return readers

    def clean(self, raw: Mapping[str, pl.DataFrame]) -> Mapping[str, CleanTable]:
        from domains.coffee.clean import clean_tables

        return clean_tables(raw, self.config.cleaning, self.config.production)

    def clean_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        return clean_schemas(self.config.cleaning)

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

        return studies(clean, self.config.market_analysis, self.config.production)

    def figures(self, tables: Mapping[str, pl.DataFrame]) -> Mapping[str, Figure]:
        from domains.coffee.analysis import figures

        return figures(tables, self.config.market_analysis)


def hooks(model: str) -> ModelHooks:
    """The code behind one of the YAML's models; a model with none is a config error."""
    if model not in MODELS:
        raise ValueError(f"Coffee has no code for model '{model}'; it has {list(MODELS)}")
    return MODELS[model]
