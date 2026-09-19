"""Runtime settings (environment) and domain config (YAML), both validated with pydantic."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator
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
    # CSV inside the archive, when the download is a ZIP.
    member: str | None = None
    # Literal strings the upstream uses for missing values (e.g. R writes "NA").
    null_values: list[str] = []

    @model_validator(mode="after")
    def _zip_needs_member(self) -> Self:
        if self.filename.endswith(".zip") and self.member is None:
            raise ValueError(f"'{self.filename}' is a ZIP: set `member` to the CSV inside it")
        return self


class DomainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sources: dict[str, SourceConfig]


def load_domain_config(domain: str, configs_dir: Path = CONFIGS_DIR) -> DomainConfig:
    path = configs_dir / f"{domain}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No config for domain '{domain}' at {path}")
    return DomainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
