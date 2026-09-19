"""Runtime settings (environment) and domain config (YAML), both validated with pydantic."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class Settings(BaseSettings):
    """Machine-specific values: read from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="COFFEE_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")


class SourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    filename: str


class DomainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sources: dict[str, SourceConfig]


def load_domain_config(domain: str, configs_dir: Path = CONFIGS_DIR) -> DomainConfig:
    path = configs_dir / f"{domain}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No config for domain '{domain}' at {path}")
    return DomainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
