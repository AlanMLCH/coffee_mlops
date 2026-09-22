"""Runtime settings (environment) and the generic half of a domain's config (YAML).

A domain's YAML has two kinds of section. The ones every domain has - its file
downloads, what one item is, the model, training and analysis - are defined here,
because the core runs them. The ones only one domain has (an API's paging, a cleaning
vocabulary) are defined by that domain, which extends `DomainConfig` with them; pydantic
still refuses any key that nobody declared.
"""

from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Machine-specific values: read from `MLOPS_*` environment variables or `.env`.

    A domain's credentials are not here: they belong to the domain that uses them, which
    reads them under its own prefix.
    """

    model_config = SettingsConfigDict(env_prefix="MLOPS_", env_file=".env", extra="ignore")

    # The domain a command runs when none is named. Unset, a lone installed domain is used.
    domain: str | None = None
    data_dir: Path = Path("data")
    # The MLflow server from docker-compose. Tests point it at a throwaway SQLite file.
    mlflow_tracking_uri: str = "http://localhost:5000"
    # Where the API keeps its copy of the champion. Set it outside data_dir when the
    # data is mounted read-only. Defaults to <data_dir>/<domain>/model_cache.
    model_cache_dir: Path | None = None
    # Complete partitions kept per table when pruning; history explains past predictions.
    keep_partitions: int = 3


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
    """A file the domain downloads as it is: a table, or a map layer inside an archive."""

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
            raise ValueError(f"'{self.filename}' is a ZIP: set `member` to the file inside it")
        return self


class ItemsConfig(BaseModel):
    """What one item of the catalog is, in the domain's own vocabulary.

    This is what lets the model pipeline run on any domain: it never names what an
    item is, only "the id column" and "the time column" named here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    noun: str  # names the model's tables: <noun>_features and <noun>_predictions
    table: str  # the clean table that holds one row per item
    id: str
    # When the item was measured. It orders the temporal split, and it dates the
    # context an item may see: nothing published after it.
    time: str
    period: str  # the column that separates periods, for drift and residuals

    @property
    def features_table(self) -> str:
        return f"{self.noun}_features"

    @property
    def predictions_table(self) -> str:
        return f"{self.noun}_predictions"

    @property
    def keys(self) -> list[str]:
        """Carried through every model table, for joins and for splitting."""
        return [self.id, self.period, self.time]


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


class TargetBands(BaseModel):
    """Ranges of the target that mean something to a reader, for reporting error by band.

    Deciles would be generic and useless: whoever reads the error cares about the ranges
    their own field uses, so the domain names them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Inner boundaries, ascending. A value equal to an edge belongs to the band above it.
    edges: list[float]
    labels: list[str]  # one more than the edges

    @model_validator(mode="after")
    def _one_label_per_band(self) -> Self:
        if len(self.labels) != len(self.edges) + 1:
            raise ValueError(f"{len(self.edges)} edges make {len(self.edges) + 1} bands")
        if self.edges != sorted(self.edges):
            raise ValueError("Band edges must be ascending")
        return self


class AnalysisConfig(BaseModel):
    """The studies every domain gets. A domain adds its own in its own config section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Groups and levels smaller than this are not reported: the numbers would be noise.
    min_rows: int
    permutation_repeats: int
    target_bands: TargetBands
    # Figures copied into docs/figures/ (committed, rendered on GitHub).
    published_figures: list[str]


class DomainConfig(BaseModel):
    """The sections the core runs. A domain subclasses this to add its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sources: dict[str, SourceConfig]
    items: ItemsConfig
    model: ModelSpec
    training: TrainingConfig
    analysis: AnalysisConfig


def load_config[Config: DomainConfig](path: Path, model: type[Config]) -> Config:
    """Read a domain's YAML and validate it against that domain's config model."""
    if not path.is_file():
        raise FileNotFoundError(f"No domain config at {path}")
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
