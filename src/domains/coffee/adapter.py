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

from domains.coffee.auction import AUCTION_CONTEXT, add_lot_market
from domains.coffee.buyers import add_household_traits
from domains.coffee.config import CoffeeConfig, CoffeeCredentials
from domains.coffee.features import (
    CONTEXT_TABLE,
    ORIGINS_TABLE,
    add_coffee_origin,
    add_market_context,
)
from domains.coffee.forecast import PRICES_TABLE, add_price_history
from domains.coffee.outlook import add_price_outlook
from domains.coffee.request import (
    AuctionLot,
    Household,
    Lot,
    Offer,
    Place,
    PriceMonth,
    PriceOutlook,
    ShelfItem,
    Zone,
)
from domains.coffee.schemas import (
    ENIGH_HOUSEHOLDS_RAW,
    ENIGH_SPENDING_RAW,
    RAW_SCHEMAS,
    borough_profile_schema,
    clean_schemas,
    household_coffee_schema,
    intercensal_schema,
)
from domains.coffee.shelf import SHELF_CONTEXT, add_shelf_context
from domains.coffee.shop_kind import add_place_traits
from domains.coffee.zone_profile import ZONE_CONTEXT, add_zone_profile
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
    # How many coffee shops a zone of the city has, for what it is: its context is the
    # census, the registers and the stations it is looked up in.
    "zones": ModelHooks(
        context_tables=ZONE_CONTEXT,
        enrich=add_zone_profile,
        request=Zone,
    ),
    # Where green coffee could be in 3, 6 or 12 months: the history is its context.
    "green_range": ModelHooks(
        context_tables=(PRICES_TABLE,),
        enrich=add_price_outlook,
        request=PriceOutlook,
    ),
    # What a jar should cost on a shelf, in today's pesos: the index is its context.
    "shelf_price": ModelHooks(
        context_tables=SHELF_CONTEXT,
        enrich=add_shelf_context,
        request=ShelfItem,
    ),
    # Whether a place is a coffee shop: its zone is its context.
    "shop_kind": ModelHooks(
        context_tables=ZONE_CONTEXT,
        enrich=add_place_traits,
        request=Place,
    ),
    # What a lot fetches at auction: the other lots of its year, and its market.
    "auction": ModelHooks(
        context_tables=AUCTION_CONTEXT,
        enrich=add_lot_market,
        request=AuctionLot,
    ),
    # Whether a household buys coffee: it needs nothing beyond itself.
    "households": ModelHooks(
        context_tables=(),
        enrich=lambda items, context: add_household_traits(items),
        request=Household,
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
            "INEGI indicators token (COFFEE_INPC_TOKEN)": keys.inpc_token,
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
        profile, survey = self.config.borough_profile, self.config.household_spending
        return {
            **RAW_SCHEMAS,
            **dict.fromkeys(closed, RAW_SCHEMAS["profeco_prices"]),
            profile.source: intercensal_schema(profile),
            survey.spending: ENIGH_SPENDING_RAW,
            survey.households: ENIGH_HOUSEHOLDS_RAW,
        }

    def json_readers(self) -> Mapping[str, JsonReader]:
        from domains.coffee.sources import denue, fas, inpc, overpass, roasters

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
        if config.inpc is not None:
            readers[config.inpc.name] = inpc.to_frame
        if config.roasters is not None:
            readers[config.roasters.name] = roasters.to_frame
        return readers

    def file_readers(self) -> Mapping[str, FileReader]:
        from domains.coffee.sources.census_zones import read_census_zones
        from domains.coffee.sources.coe import read_competition
        from domains.coffee.sources.enigh import read_households, read_spending
        from domains.coffee.sources.faostat import read_producer_prices
        from domains.coffee.sources.ico import read_indicator_prices
        from domains.coffee.sources.profeco import read_shelf_prices

        shelves, farmers = self.config.consumer_prices, self.config.producer_prices
        readers: dict[str, FileReader] = {
            "ico_prices": read_indicator_prices,
            "cup_of_excellence": read_competition,
        }
        # CoffeeConfig refuses a producer price source without a member.
        readers[farmers.source] = partial(read_producer_prices,
                                          member=str(self.config.sources[farmers.source].member),
                                          item_code=farmers.item_code)  # fmt: skip
        # CoffeeConfig refuses a zones census without a member too.
        zones = self.config.census_zones.census
        readers[zones] = partial(read_census_zones, member=str(self.config.sources[zones].member))
        # And a household survey's files without one.
        survey, sources = self.config.household_spending, self.config.sources
        readers[survey.spending] = partial(read_spending,
                                           member=str(sources[survey.spending].member),
                                           codes=list(survey.products))  # fmt: skip
        readers[survey.households] = partial(
            read_households, member=str(sources[survey.households].member)
        )
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
            config.transit,
            config.census_zones,
            config.household_spending,
            read_at,
        )

    def clean_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        profile = borough_profile_schema(self.config.borough_profile)
        households = household_coffee_schema(self.config.household_spending)
        return {
            **clean_schemas(self.config.cleaning),
            "borough_profile": profile,
            "household_coffee": households,
        }

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
            config.household_spending,
        )

    def figures(self, tables: Mapping[str, pl.DataFrame]) -> Mapping[str, Figure]:
        from domains.coffee.analysis import figures

        survey = self.config.household_spending
        return figures(tables, self.config.market_analysis, survey.states[survey.city])


def hooks(model: str) -> ModelHooks:
    """The code behind one of the YAML's models; a model with none is a config error."""
    if model not in MODELS:
        raise ValueError(f"Coffee has no code for model '{model}'; it has {list(MODELS)}")
    return MODELS[model]
