"""Runtime settings (environment) and the generic half of a domain's config (YAML).

A domain's YAML has two kinds of section. The ones every domain has - its file
downloads, its corpus, its models (what one item is, what is predicted, how it is
trained) and its analysis - are defined here, because the core runs them. The ones
only one domain has (an API's paging, a cleaning vocabulary) are defined by that
domain, which extends `DomainConfig` with them; pydantic still refuses any key that
nobody declared.
"""

from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator
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
    # Ollama, serving the local models natively (it sees the GPU; a container would not).
    # 127.0.0.1, not localhost: on Windows localhost resolves to IPv6 first, and a service
    # listening on IPv4 only makes every new connection wait for that attempt to time out.
    ollama_url: str = "http://127.0.0.1:11434"
    # Qdrant, from docker-compose (profile `ai`), published on 127.0.0.1 only.
    qdrant_url: str = "http://127.0.0.1:6333"


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
    # Character set of a table. A government CSV in Latin-1 declares it nowhere, and
    # read as UTF-8 it fails on the first accented name rather than mangling it.
    encoding: str = "utf-8"
    # Set when the member is a map layer rather than a table.
    spatial: SpatialConfig | None = None

    @model_validator(mode="after")
    def _zip_needs_member(self) -> Self:
        if self.filename.endswith(".zip") and self.member is None:
            raise ValueError(f"'{self.filename}' is a ZIP: set `member` to the file inside it")
        return self


class DocumentConfig(BaseModel):
    """One document of the domain's corpus: where it comes from, and what it is.

    Text the agent explains from, never figures: a PDF's tables come out of extraction
    scrambled, and an answer built from them is confidently wrong. Numbers are answered
    from the tables, which is what the SQL tool is for.

    The metadata is not decoration: it travels with every chunk, so an answer can say who
    published it, when, and under what licence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str  # its raw source folder
    title: str
    publisher: str
    # The edition's year, where it states one: a catalogue that is revised silently does
    # not, and a guessed year in a citation is worse than none.
    year: int | None = None
    # Where it came from: fetched from here, and cited in an answer.
    url: HttpUrl
    license: str
    language: str  # of the text as published, ISO 639-1
    topics: list[str] = Field(min_length=1)  # the domain's own vocabulary
    format: Literal["pdf", "jats"]  # JATS is the XML article format Europe PMC serves
    # Some publishers answer 403 to anything that is not a browser, licence
    # notwithstanding. Those are fetched by hand into <data_dir>/inbox/documents/ and
    # named here: a refusal is respected, never worked around.
    inbox: str | None = None


class TopicConfig(BaseModel):
    """One subject of the corpus, in the domain's own words.

    The terms are what a chunk is tagged by and a question routed by, and what a glossary
    joins to the tables' closed vocabularies (a process named in a text and in a column).
    A term matches as a whole word or phrase, in its inflections - "roast" finds roasts,
    roasted, roasting and roaster - and never inside another word.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str  # what a document under this topic answers
    terms: list[str] = Field(min_length=1)


class ChunkingConfig(BaseModel):
    """How long a chunk is: a retrieval setting, to be tuned through the gate like a model.

    In characters, not tokens: cutting needs no tokenizer, and English text runs at about
    four characters a token.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_chars: int = Field(gt=0)
    # Carried from the end of one chunk into the next, in whole sentences, so a passage
    # cut at a boundary is still found whole in one of the two.
    overlap_chars: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_is_shorter_than_a_chunk(self) -> Self:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be shorter than max_chars")
        return self


class CorpusConfig(BaseModel):
    """What the corpus is filed under and how it is cut.

    The topics are the domain's words; that they exist, and that every document is filed
    under some, is the core's rule, because the core tags each chunk and routes each
    question by them. Metadata, not folders: almost no document is about one subject.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topics: dict[str, TopicConfig] = Field(min_length=1)
    chunking: ChunkingConfig


# The clean tables the core builds from a corpus, next to the domain's own.
DOCUMENTS_TABLE = "documents"
CHUNKS_TABLE = "document_chunks"


class ItemsConfig(BaseModel):
    """What one item of a model is, in the domain's own vocabulary.

    This is what lets the model pipeline run on any domain: it never names what an
    item is, only "the id column" and "the time column" named here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str  # the clean table that holds one row per item
    id: str
    # When the item was measured. It orders a temporal split, and it dates the context
    # an item may see: nothing published after it.
    time: str
    period: str  # the column that separates periods, for drift and residuals


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


class TemporalSplit(BaseModel):
    """Past trains, future evaluates: for items with a time axis worth predicting across."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["temporal"]
    test_from: date
    # Items used to estimate the level shift when simulating a deployed recalibration.
    # Only a temporal split has a "next period" to recalibrate on.
    recalibration_window: int


class GroupSplit(BaseModel):
    """Whole groups go to one side: for items that come in families.

    Several items of one group (sizes of one product, say) share almost everything; with
    some in train and the rest in test, the model would be scored on what it memorised.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["group"]
    column: str  # items sharing a value never straddle train and test
    test_share: float = Field(gt=0, lt=1)  # of the groups, not of the items


Split = Annotated[TemporalSplit | GroupSplit, Field(discriminator="kind")]


# What the tuner may try, as (low, high) per parameter. The defaults suit a few thousand
# rows; a model with a few hundred says so, because a search that can reach 800 trees can
# also reach a learning rate so low that the model never leaves the mean.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "learning_rate": (0.01, 0.2),
    "n_estimators": (50, 800),
    "num_leaves": (4, 64),
    "min_child_samples": (5, 60),
    "reg_lambda": (1e-3, 10.0),
    "colsample_bytree": (0.5, 1.0),
    # How many rows a category needs to exist on its own; above that it is grouped as
    # "infrequent", which on a small table quietly deletes the categorical features.
    "min_frequency": (2, 30),
}


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Split
    # Bounds to narrow, merged over the defaults above.
    search_space: dict[str, tuple[float, float]] = {}
    cv_folds: int  # folds inside the training split, of the same kind as the split
    # How many times those folds are drawn. One pass over a few hundred rows is a noisy
    # thing to choose hyperparameters by: the tuner ends up ranking draws, not models.
    # Only a group split can repeat them (time has one order).
    cv_repeats: int = Field(default=1, ge=1)
    trials: int
    seed: int
    baseline_group: str
    bootstrap_resamples: int
    min_probability_better: float
    stratify_by: str
    min_group_size: int
    registered_model: str

    @model_validator(mode="after")
    def _search_space_is_known_and_ordered(self) -> Self:
        unknown = sorted(set(self.search_space) - set(SEARCH_SPACE))
        if unknown:
            raise ValueError(f"Nothing to tune called {unknown}; there is {sorted(SEARCH_SPACE)}")
        backwards = sorted(name for name, (low, high) in self.search_space.items() if low >= high)
        if backwards:
            raise ValueError(f"Search bounds must run from low to high: {backwards}")
        return self

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return SEARCH_SPACE | self.search_space


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


class ModelConfig(BaseModel):
    """One model of a domain: its items, what it predicts, and how it is trained.

    A domain can have several - one per question it asks of its data - and each gets its
    own tables, MLflow experiment, registered model, API route and studies, all named
    after it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str  # names its tables: <name>_features and <name>_predictions
    items: ItemsConfig
    spec: ModelSpec
    training: TrainingConfig
    # Error is reported by the target ranges the domain's readers use, not by deciles.
    target_bands: TargetBands

    @model_validator(mode="after")
    def _group_is_its_own_column(self) -> Self:
        split, items = self.training.split, self.items
        taken = {items.id, items.period, items.time, self.spec.target, *self.spec.features}
        if isinstance(split, GroupSplit) and split.column in taken:
            raise ValueError(
                f"{self.name}: split.column '{split.column}' must be a column of its own, "
                "not a key, a feature or the target"
            )
        return self

    @property
    def features_table(self) -> str:
        return f"{self.name}_features"

    @property
    def predictions_table(self) -> str:
        return f"{self.name}_predictions"

    @property
    def keys(self) -> list[str]:
        """Carried through every model table, for joins and for splitting."""
        items, split = self.items, self.training.split
        grouped = [split.column] if isinstance(split, GroupSplit) else []
        return [items.id, items.period, items.time, *grouped]


class AnalysisConfig(BaseModel):
    """The studies every domain gets. A domain adds its own in its own config section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Groups and levels smaller than this are not reported: the numbers would be noise.
    min_rows: int
    permutation_repeats: int
    # Figures copied into docs/figures/ (committed, rendered on GitHub). A model's own
    # figures are named after it: <model>_<figure>.
    published_figures: list[str]


class DomainConfig(BaseModel):
    """The sections the core runs. A domain subclasses this to add its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sources: dict[str, SourceConfig]
    documents: list[DocumentConfig] = []  # the corpus; empty until a domain has one
    corpus: CorpusConfig | None = None  # required once there are documents
    models: list[ModelConfig] = Field(min_length=1)
    analysis: AnalysisConfig

    @model_validator(mode="after")
    def _documents_are_named_once(self) -> Self:
        names = [document.name for document in self.documents]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Document names must be unique; repeated: {repeated}")
        return self

    @model_validator(mode="after")
    def _documents_are_filed_under_known_topics(self) -> Self:
        """A topic nobody declared would route nothing and be found by nobody."""
        if self.documents and self.corpus is None:
            raise ValueError("Documents need a `corpus:` section: their topics and how to cut them")
        known = self.corpus.topics if self.corpus else {}
        unknown = {
            f"{document.name}: {topic}"
            for document in self.documents
            for topic in document.topics
            if topic not in known
        }
        if unknown:
            raise ValueError(f"Documents filed under topics corpus.topics lacks: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def _models_are_named_once(self) -> Self:
        names = [model.name for model in self.models]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Model names must be unique; repeated: {repeated}")
        return self

    @property
    def corpus_tables(self) -> tuple[str, ...]:
        """The clean tables the core builds from the corpus; none for a domain without one."""
        return (DOCUMENTS_TABLE, CHUNKS_TABLE) if self.documents else ()

    def model_named(self, name: str) -> ModelConfig:
        """The model called `name`, or an error that lists the ones there are."""
        for model in self.models:
            if model.name == name:
                return model
        raise ValueError(f"No model '{name}'; {self.name} has {[m.name for m in self.models]}")


def load_config[Config: DomainConfig](path: Path, model: type[Config]) -> Config:
    """Read a domain's YAML and validate it against that domain's config model."""
    if not path.is_file():
        raise FileNotFoundError(f"No domain config at {path}")
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
