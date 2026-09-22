"""The coffee domain's own config sections and credentials.

The core's `DomainConfig` covers what every domain has. Coffee adds three API sources
that each need code (DENUE pages, Overpass takes a query, FAS wants a key and walks
years), a cleaning vocabulary for two CQI snapshots that disagree about spelling, and
the market studies that only make sense for a commodity with a world balance.
"""

from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlops_core.config import DomainConfig


class CoffeeCredentials(BaseSettings):
    """The domain's keys, under its own prefix: `COFFEE_DENUE_TOKEN`, `COFFEE_USDA_FAS_API_KEY`.

    SecretStr so a value cannot leak through a repr, a log line or a traceback: printing
    one shows `SecretStr('**********')`, and reading it takes an explicit
    `.get_secret_value()`. They live in `.env`, which is gitignored.
    """

    model_config = SettingsConfigDict(env_prefix="COFFEE_", env_file=".env", extra="ignore")

    denue_token: SecretStr | None = None  # INEGI, free
    usda_fas_api_key: SecretStr | None = None  # USDA FAS Open Data, free


class DenueConfig(BaseModel):
    """The DENUE inventory to pull: one activity class in one state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str  # the raw source name, and therefore its folder
    base_url: str
    entity: str  # INEGI state code; "09" is Mexico City
    activity_class: str  # SCIAN class
    page_size: int
    filename: str
    rate_limit_seconds: float
    # How long a cached page stays true. Required on purpose: the default that needs no
    # thought is "forever", which silently turns a live register into a snapshot.
    cache_hours: float


class OverpassConfig(BaseModel):
    """The OpenStreetMap inventory to pull: some amenity tags inside one administrative area."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    base_url: str
    area_iso: str  # ISO 3166-2 code of the area; "MX-CMX" is Mexico City
    amenities: list[str]  # OSM `amenity` values, e.g. ["cafe", "ice_cream"]
    filename: str
    # Overpass' own budget for the query. Must stay under the HTTP read timeout so the
    # server's explanation arrives before the client gives up without one.
    timeout_s: int
    rate_limit_seconds: float
    cache_hours: float  # see DenueConfig


class FasConfig(BaseModel):
    """The USDA FAS balance to pull: one commodity, every market year from `first_year`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    base_url: str
    commodity_code: str  # PSD commodity; "0711100" is "Coffee, Green"
    first_year: int
    filename: str
    rate_limit_seconds: float
    cache_hours: float  # see DenueConfig


class ShopKindRule(BaseModel):
    """One kind of place, recognised by a pattern in its name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    # A regex over the upper-cased, accent-stripped name.
    pattern: str


class RegisterMatchConfig(BaseModel):
    """When an entry in one register and an entry in another are the same place."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    radius_m: float
    min_name_similarity: float  # Jaro-Winkler, 0-1


class CleaningConfig(BaseModel):
    """Rules for the clean layer. Vocabularies are closed: an unseen label stops the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    altitude_m: tuple[float, float]
    country_aliases: dict[str, str]
    processing_methods: dict[str, str]
    colors: dict[str, str | None]
    # Ordered: the first rule whose pattern matches a name decides the kind.
    shop_kinds: list[ShopKindRule]
    # OSM amenity tag -> kind. OSM's mappers already said what the place is.
    osm_kinds: dict[str, str]
    register_match: RegisterMatchConfig

    @property
    def kinds(self) -> list[str]:
        """Every kind a place can have, including the two no rule assigns."""
        named = [rule.kind for rule in self.shop_kinds]
        return [*dict.fromkeys([*named, *self.osm_kinds.values(), UNCLASSIFIED])]


UNCLASSIFIED = "unclassified"  # named, but the name says nothing any rule recognises


class MarketAnalysisConfig(BaseModel):
    """Which slice of the world market the coffee-only studies summarise."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    market_year: int
    top_countries: int
    # The country the domestic-market view follows through time.
    spotlight_country: str
    history_since: int


class CoffeeConfig(DomainConfig):
    """Everything in `config.yaml`: the core's sections plus coffee's own."""

    denue: DenueConfig | None = None
    overpass: OverpassConfig | None = None
    fas: FasConfig | None = None
    cleaning: CleaningConfig
    market_analysis: MarketAnalysisConfig
