"""Runtime settings (environment) and domain config (YAML), both validated with pydantic."""

from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class Settings(BaseSettings):
    """Machine-specific values: read from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="COFFEE_", env_file=".env", extra="ignore")

    domain: str = "coffee"
    data_dir: Path = Path("data")
    # The MLflow server from docker-compose. Tests point it at a throwaway SQLite file.
    mlflow_tracking_uri: str = "http://localhost:5000"
    # Where the API keeps its copy of the champion. Set it outside data_dir when the
    # data is mounted read-only. Defaults to <data_dir>/<domain>/model_cache.
    model_cache_dir: Path | None = None
    # Complete partitions kept per table when pruning; history explains past predictions.
    keep_partitions: int = 3

    # Credentials for the stage 2 sources. SecretStr so the value cannot leak through a
    # repr, a log line or a traceback: printing one shows `SecretStr('**********')`, and
    # reading it takes an explicit `.get_secret_value()`. They live in `.env`, which is
    # gitignored; `.env.example` documents them.
    denue_token: SecretStr | None = None  # INEGI, free
    usda_fas_api_key: SecretStr | None = None  # USDA FAS Open Data, free


class FasConfig(BaseModel):
    """The USDA FAS balance to pull: one commodity, every market year from `first_year`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str  # the raw source name, and therefore its folder
    base_url: str
    commodity_code: str  # PSD commodity; "0711100" is "Coffee, Green"
    first_year: int
    filename: str
    rate_limit_seconds: float
    cache_hours: float  # see DenueConfig


class SpatialConfig(BaseModel):
    """How to read a geospatial layer, for a source whose download is not a table.

    Everything here is stated rather than discovered, because the file does not say it
    reliably: a shapefile's DBF declares no character set, and its `.prj` is advisory.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoding: str  # character set of the attribute table (DBF)
    crs: str  # the layer's own coordinate system; geometry is reprojected to WGS84
    id_column: str
    name_column: str
    # How many features the layer must have. An official boundary set has a known
    # number of areas, so a different count is a changed upstream, not a surprise.
    expected_features: int


class SourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    filename: str
    # The file inside the archive, when the download is a ZIP: a CSV, or the layer of
    # a geospatial dataset when `spatial` is set.
    member: str | None = None
    # Literal strings the upstream uses for missing values (e.g. R writes "NA").
    null_values: list[str] = []
    # Set when the member is a map layer rather than a table.
    spatial: SpatialConfig | None = None

    @model_validator(mode="after")
    def _zip_needs_member(self) -> Self:
        if self.filename.endswith(".zip") and self.member is None:
            raise ValueError(f"'{self.filename}' is a ZIP: set `member` to the CSV inside it")
        return self


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
    """The OpenStreetMap inventory to pull: one amenity tag inside one administrative area."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str  # the raw source name, and therefore its folder
    base_url: str
    area_iso: str  # ISO 3166-2 code of the area; "MX-CMX" is Mexico City
    amenity: str  # OSM `amenity` value, e.g. "cafe"
    filename: str
    # Overpass' own budget for the query. Must stay under the HTTP read timeout so the
    # server's explanation arrives before the client gives up without one.
    timeout_s: int
    rate_limit_seconds: float
    cache_hours: float  # see DenueConfig


class CleaningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    altitude_m: tuple[float, float]
    country_aliases: dict[str, str]
    processing_methods: dict[str, str]
    colors: dict[str, str | None]


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    categorical: list[str]
    numeric: list[str]
    # Columns that must never be features (target leakage).
    leakage: list[str]

    @property
    def features(self) -> list[str]:
        return [*self.categorical, *self.numeric]

    @model_validator(mode="after")
    def _no_leakage(self) -> Self:
        leaked = set(self.features) & {*self.leakage, self.target}
        if leaked:
            raise ValueError(f"Leaking columns declared as features: {sorted(leaked)}")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    test_from: date
    cv_folds: int
    trials: int
    seed: int
    baseline_group: str
    bootstrap_resamples: int
    min_probability_better: float
    stratify_by: str
    min_group_size: int
    recalibration_window: int
    registered_model: str


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    period_column: str
    min_rows: int
    market_year: int
    top_countries: int
    spotlight_country: str
    history_since: int
    permutation_repeats: int
    published_figures: list[str]


class DomainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sources: dict[str, SourceConfig]
    # API sources are not plain downloads -- they page, they carry credentials, they
    # speak their own query language -- so each is configured apart from the file
    # sources until the contract is extracted (end of stage 2).
    denue: DenueConfig | None = None
    overpass: OverpassConfig | None = None
    fas: FasConfig | None = None
    cleaning: CleaningConfig
    model: ModelSpec
    training: TrainingConfig
    analysis: AnalysisConfig


def load_domain_config(domain: str, configs_dir: Path = CONFIGS_DIR) -> DomainConfig:
    path = configs_dir / f"{domain}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No config for domain '{domain}' at {path}")
    return DomainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
