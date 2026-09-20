"""Runtime settings (environment) and domain config (YAML), both validated with pydantic."""

from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator
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


class SourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    filename: str
    # CSV inside the archive, when the download is a ZIP.
    member: str | None = None
    # Literal strings the upstream uses for missing values (e.g. R writes "NA").
    null_values: list[str] = []

    @model_validator(mode="after")
    def _zip_needs_member(self) -> Self:
        if self.filename.endswith(".zip") and self.member is None:
            raise ValueError(f"'{self.filename}' is a ZIP: set `member` to the CSV inside it")
        return self


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
    cleaning: CleaningConfig
    model: ModelSpec
    training: TrainingConfig
    analysis: AnalysisConfig


def load_domain_config(domain: str, configs_dir: Path = CONFIGS_DIR) -> DomainConfig:
    path = configs_dir / f"{domain}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No config for domain '{domain}' at {path}")
    return DomainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
