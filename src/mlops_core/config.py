"""Runtime settings (environment) and the generic half of a domain's config (YAML).

A domain's YAML has two kinds of section. The ones every domain has - its file
downloads, its corpus, its models (what one item is, what is predicted, how it is
trained) and its analysis - are defined here, because the core runs them. The ones
only one domain has (an API's paging, a cleaning vocabulary) are defined by that
domain, which extends `DomainConfig` with them; pydantic still refuses any key that
nobody declared.
"""

import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator
from pydantic_core import to_jsonable_python
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
    # The prediction API, from docker-compose (profile `api`): the agent's prediction tool.
    api_url: str = "http://127.0.0.1:8000"
    # Whether Dagster's schedule and sensors start on by themselves. Off: the project runs
    # by hand, and opening the orchestrator's UI starts nothing. A deployment meant to keep
    # its own history (the ICO's month, the shops' catalogues) sets MLOPS_AUTOMATE=true.
    automate: bool = False
    # The hosted models the agent may ask before its local one (`rag.providers`), each
    # used only when its key is set. No file, or no key: the local model alone.
    providers_file: Path = Path("providers.yaml")
    # Ask only one: "local", or a provider's name. Unset, the whole chain answers.
    generator: str | None = None
    # Providers that bill (Anthropic, OpenAI) join the chain only when this is true.
    paid_providers: bool = False
    # Keep every reply by its prompt, so a prompt asked before costs nothing
    # (`rag.providers.CachedGenerator`). Evaluations never use it: they measure the model.
    reply_cache: bool = True
    # Where `make explore` serves the explorer: the MCP server's results link to it.
    explore_url: str = "http://localhost:8502"
    # A snapshot `mlops export` wrote - a URL, such as a release asset, or a path: set, the
    # explorer runs as a showcase, from it alone, without the agent.
    showcase: str | None = None


def unread_settings(names: Iterable[str], domain: str) -> dict[str, str]:
    """Variables that look like settings but that no setting reads, each with why.

    Pydantic ignores an unknown variable in silence, so a name that is off by a prefix
    (`<DOMAIN>_DATA_DIR`, from before the core had its own) or by a letter leaves its
    setting at the default - which can look like it worked. Two cases: an `MLOPS_*` name
    that is no setting, and a core setting written under the domain's prefix.
    """
    fields = {name.upper() for name in Settings.model_fields}
    prefix = f"{domain.upper()}_"
    found = {}
    for name in names:
        upper = name.upper()
        if upper.startswith("MLOPS_") and upper.removeprefix("MLOPS_") not in fields:
            found[name] = "no setting has this name"
        elif upper.startswith(prefix) and upper.removeprefix(prefix) in fields:
            found[name] = f"not read: the setting is MLOPS_{upper.removeprefix(prefix)}"
    return found


def env_file_names(path: Path) -> list[str]:
    """The variable names a `.env` file sets, values never read into anything."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        line.split("=", 1)[0].strip()
        for line in lines
        if "=" in line and not line.lstrip().startswith("#")
    ]


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


YEAR = "{year}"
_ESCAPED_YEAR = "%7Byear%7D"


class YearRange(BaseModel):
    """The years a statistic has a file for, both included."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first: int
    last: int
    # Years between them the publisher has no file for - an edition never held - so not
    # a download that failed. The last year is never one: it is the edition in course.
    missing: list[int] = []

    @model_validator(mode="after")
    def _in_order(self) -> Self:
        if self.first > self.last:
            raise ValueError(f"years run from first to last: {self.first} > {self.last}")
        outside = [year for year in self.missing if not self.first <= year < self.last]
        if outside:
            raise ValueError(f"missing years {outside} are not before {self.last} in the range")
        return self


class SourceConfig(BaseModel):
    """A file the domain downloads as it is: a table (CSV, or a workbook's sheet), a map
    layer inside an archive, or a file only the domain can read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    # When set, `url` is the page that publishes the file, and the file is the first link
    # on it this pattern matches: a release whose address carries an id per edition, on
    # a page whose address does not.
    link: str | None = None
    # The same, matched against what the link says rather than where it points: a page
    # whose addresses are opaque tokens (`file.php?t=9d62...`) names its files only in
    # their text.
    link_text: str | None = None
    filename: str
    # The file inside the archive, when the download is a ZIP: a CSV, or the layer of
    # a geospatial dataset when `spatial` is set. For an archive of many tables that a
    # domain's own reader reads, the folder they are in.
    member: str | None = None
    # Literal strings the upstream uses for missing values (e.g. R writes "NA").
    null_values: list[str] = []
    # Character set of a table. A government CSV in Latin-1 declares it nowhere, and
    # read as UTF-8 it fails on the first accented name rather than mangling it.
    encoding: str = "utf-8"
    # Set when the member is a map layer rather than a table.
    spatial: SpatialConfig | None = None
    # A workbook's sheet, the row holding its column names (counted from 0), and the rows
    # right under it that are not data - a row of units, say.
    sheet: str | None = None
    header_row: int = Field(0, ge=0)
    skip_rows: int = Field(0, ge=0)
    # How long a download stays fresh: a run within this many hours of the last check
    # does not download again. Unset, every run downloads (and stores only a change). A
    # scheduled daily run would otherwise fetch an 83 MB boundary file that last
    # changed in 2020, every day.
    refresh_hours: float | None = Field(None, gt=0)
    # A statistic published one file a year at the same address but for the year, written
    # `{year}` in `url` and `filename`: each year is its own download, a closed year -
    # every one but the last - is downloaded once and kept, and the years are read as
    # one table. The next year closes by moving `last`.
    years: YearRange | None = None
    # A column's older header -> its name now, for a file whose headers changed between
    # editions ("Precio" before 2021, "Preciomediorural" since). Applied when read.
    renamed: dict[str, str] = {}
    # A thousands separator some cells use ("10,271" among thousands of "10271"): taken
    # out when read, from a cell that is a number and nothing else, so a name keeps it.
    thousands: str | None = Field(None, min_length=1, max_length=1)
    # The seconds its host's robots.txt asks between requests (`Crawl-delay`), read by
    # hand like the rest of this entry. Two downloads from that host in one run wait it
    # out between them; one download alone waits for nothing.
    crawl_delay: float | None = Field(None, gt=0)

    def address(self) -> str:
        """`url` as written: the URL type escapes the braces of `{year}` in a path (not in
        a query), and a year in the path is the address of a page a year."""
        return str(self.url).replace(_ESCAPED_YEAR, YEAR)

    def editions(self) -> list[tuple[int, str, str]]:
        """Each year's (year, url, filename); none for a source that is one file."""
        if self.years is None:
            return []
        return [
            (year, self.address().replace(YEAR, str(year)), self.filename.replace(YEAR, str(year)))
            for year in range(self.years.first, self.years.last + 1)
            if year not in self.years.missing
        ]

    @model_validator(mode="after")
    def _a_year_is_written_where_it_changes(self) -> Self:
        named = YEAR in self.address() and YEAR in self.filename
        if (self.years is not None) != named:
            raise ValueError(
                f"`years` and {YEAR} in both `url` and `filename` go together: '{self.filename}'"
            )
        if self.years is not None and (self.link or self.link_text):
            raise ValueError("A file a year is addressed by its year, not found by a link")
        return self

    @model_validator(mode="after")
    def _zip_needs_member(self) -> Self:
        if self.filename.endswith(".zip") and self.member is None:
            raise ValueError(f"'{self.filename}' is a ZIP: set `member` to the file inside it")
        return self

    @model_validator(mode="after")
    def _a_workbook_names_its_sheet(self) -> Self:
        if self.filename.endswith(".xlsx") != (self.sheet is not None):
            raise ValueError(f"'{self.filename}': a workbook names its `sheet`, and only one does")
        return self

    @model_validator(mode="after")
    def _a_link_is_found_one_way(self) -> Self:
        if self.link is not None and self.link_text is not None:
            raise ValueError("Set `link` or `link_text`, not both: a file is found one way")
        for field in ("link", "link_text"):
            pattern = getattr(self, field)
            if pattern is not None:
                try:
                    re.compile(pattern)  # a broken pattern fails when the config loads
                except re.error as broken:
                    raise ValueError(f"`{field}` is not a pattern: {broken}") from broken
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
    # How long a fetched document stays fresh (see `SourceConfig.refresh_hours`): papers
    # and catalogues are revised rarely, and the corpus weighs tens of megabytes.
    refresh_hours: float | None = Field(None, gt=0)


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
    # What stays the same thing across periods when an item is observed again - an offer
    # read week after week - where `id` names one observation. The monitor measures the
    # error only on the entities new in the newest period: the others the model has seen.
    entity: str | None = None


# What a model predicts, which decides how it learns and how its errors are counted:
# - regression: a quantity; each row's error is its absolute error.
# - count: how many of something, never negative; learned as a Poisson rate, each row
#   judged by its Poisson deviance (an absolute error would reward the median count).
# - probability: how likely a yes is, from targets of 0 and 1; learned by cross-entropy,
#   each row judged by its Brier score, the squared distance from what happened.
Task = Literal["regression", "count", "probability"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    categorical: list[str]
    numeric: list[str]
    # Columns that must never be features (target leakage).
    leakage: list[str]
    task: Task = "regression"
    # With a share, a regression also predicts a range meant to hold the truth that share
    # of the time (0.8: four times in five), from two more models at its edges' quantiles.
    # The gate then judges the range, by its interval score: width, plus a penalty for
    # every miss that grows with how far the truth fell outside.
    interval: float | None = Field(default=None, gt=0, lt=1)
    # Items whose target is not known yet are kept: scored in batch, never learned from or
    # judged on. Without it, an item without a target breaks the feature contract.
    unlabelled: bool = False
    # A numeric feature the target is the percent change of. A model learns a change
    # because a tree never predicts beyond the values it saw; whoever asks gets the answer
    # back in that feature's units as well.
    relative_to: str | None = None

    @property
    def features(self) -> list[str]:
        return [*self.categorical, *self.numeric]

    @model_validator(mode="after")
    def _no_leakage(self) -> Self:
        leaked = set(self.features) & {*self.leakage, self.target}
        if leaked:
            raise ValueError(f"Leaking columns declared as features: {sorted(leaked)}")
        return self

    @model_validator(mode="after")
    def _interval_and_change_fit_the_task(self) -> Self:
        if self.interval is not None and self.task != "regression":
            raise ValueError(f"A range is predicted around a quantity, not a {self.task}")
        if self.relative_to is not None and self.relative_to not in self.numeric:
            raise ValueError(f"relative_to must be a numeric feature: '{self.relative_to}'")
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
    # What the target is when nothing happens - 0 for a change from the period before -
    # offered as one more baseline. For a series it is the random walk: tomorrow is
    # today, which is what any forecast of a price has to beat before it is worth anything.
    baseline_constant: float | None = None
    # For a count, what it is a count *of*: a numeric feature such as the people living
    # there. The baseline is then the training rate per unit of it, times the item's own:
    # what a count model has to beat is more than the average count.
    baseline_exposure: str | None = None
    # A column whose values the gate's bootstrap resamples whole. Items that overlap in
    # time - twelve-month changes a month apart share eleven months - are not independent
    # evidence, and resampling them one by one would make the gate overconfident.
    resample_by: str | None = None
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
    # What it predicts, in the words of whoever would ask: the agent routes questions by
    # it, and a tool offering the model to another client describes itself with it.
    description: str = Field(min_length=1)
    # A request the model must answer, in the domain's request body: the API predicts it
    # whenever it loads the model and does not serve one that cannot. A stale image once
    # loaded the champion, reported itself healthy and failed every request.
    example: dict[str, Any] = Field(min_length=1)
    items: ItemsConfig
    spec: ModelSpec
    training: TrainingConfig
    # Error is reported by the target ranges the domain's readers use, not by deciles.
    target_bands: TargetBands

    @field_validator("example", mode="after")
    @classmethod
    def _example_is_json(cls, example: dict[str, Any]) -> dict[str, Any]:
        """As a request body travels: YAML reads 2026-09-01 as a date, JSON has none."""
        return to_jsonable_python(example)  # type: ignore[no-any-return]

    @model_validator(mode="after")
    def _training_names_its_columns(self) -> Self:
        cfg, spec = self.training, self.spec
        if cfg.baseline_exposure is not None and cfg.baseline_exposure not in spec.numeric:
            raise ValueError(f"{self.name}: baseline_exposure must be a numeric feature")
        if cfg.resample_by is not None and cfg.resample_by not in {*self.keys, *spec.features}:
            raise ValueError(f"{self.name}: resample_by must be a key or a feature")
        return self

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
        entity = [items.entity] if items.entity else []
        return list(dict.fromkeys([items.id, items.period, items.time, *entity, *grouped]))


class AnalysisConfig(BaseModel):
    """The studies every domain gets. A domain adds its own in its own config section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Groups and levels smaller than this are not reported: the numbers would be noise.
    min_rows: int
    permutation_repeats: int
    # Figures copied into docs/figures/ (committed, rendered on GitHub). A model's own
    # figures are named after it: <model>_<figure>.
    published_figures: list[str]


class MonitoringConfig(BaseModel):
    """When a model's newest period calls for retraining. The gate still decides whether
    what retraining produces is served."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The share of a model's features whose distribution moved - by Evidently's test for
    # each - from which it is retrained. A drifted target or an error outside the interval
    # the champion was accepted with calls for it on their own.
    drift_share: float = Field(gt=0, le=1)


class ScheduleConfig(BaseModel):
    """When the orchestrator runs the data pipeline on its own. The model pipeline needs
    no clock: it runs when the data it reads changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: str = Field(pattern=r"^\S+ \S+ \S+ \S+ \S+$")  # cron: minute hour day month weekday
    timezone: str  # IANA, e.g. Europe/Madrid: the cron is read in it


class MapView(BaseModel):
    """Where the explorer's map opens: a centre, a zoom (0 is the world), a tilt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    zoom: float = Field(ge=0, le=20)
    pitch: float = Field(0, ge=0, le=60)  # degrees: tilted, the areas' columns stand up


class AreasConfig(BaseModel):
    """The table of places a result can name, and its columns: the key and the name a
    query may use for an area, and its outline (WKB, as the core's spatial join writes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str  # layer.table, e.g. clean.<areas>
    id: str
    name: str
    boundary: str


class MapLayer(BaseModel):
    """A layer the map offers before any question is asked: a query and what it draws.
    It runs in the agent's locked session, as any question's SQL does."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: Literal["points", "areas"]  # points: latitude and longitude; areas: a key and a number
    sql: str
    description: str = ""
    unit: str = ""  # of an area's number, for the legend: "places per km²"
    # Areas of its own - a finer set of zones than the explorer's - instead of `explore.areas`.
    areas: AreasConfig | None = None


class ExploreMetric(BaseModel):
    """A headline number over the explorer: one SELECT whose first value is shown, and
    optionally one whose first column is its recent history, oldest first (a sparkline)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    sql: str
    unit: str = ""
    decimals: int = Field(0, ge=0, le=4)
    help: str = ""
    trend: str | None = None


# A column or table name as a query writes it bare: the explorer builds its SQL from these.
_IDENTIFIER = r"^[A-Za-z_]\w*$"


class ExploreDataset(BaseModel):
    """A table a person can slice in the explorer without the agent: a measure, a column to
    segment it by, another to colour it by, and filters. Every piece is a name from this
    list, so the query is assembled from the YAML, never written by a person or a model;
    it still runs in the agent's locked session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    table: str = Field(pattern=r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$")  # layer.table
    where: str | None = None  # a condition every query of it keeps: "NOT price_outlier"
    # Column alias -> an aggregate over the table: `median_mxn_per_kg: median(price_mxn_per_kg)`.
    measures: dict[str, str] = Field(min_length=1)
    dimensions: list[str] = Field(min_length=1)  # what it can be segmented and coloured by
    filters: list[str] = []  # columns a person can restrict to some of their values
    # What an empty value of a dimension means, where "(no value)" would mislead: a price
    # recorded outside every area has no area, and is not an area called "null".
    null_labels: dict[str, str] = {}

    @model_validator(mode="after")
    def _names_are_bare_identifiers(self) -> Self:
        names = [*self.measures, *self.dimensions, *self.filters]
        odd = [name for name in names if not re.match(_IDENTIFIER, name)]
        if odd:
            raise ValueError(f"{self.name}: not bare column names: {odd}")
        unknown = sorted(set(self.null_labels) - set(self.dimensions))
        if unknown:
            raise ValueError(f"{self.name}: null_labels name no dimension: {unknown}")
        return self


class ExploreFinding(BaseModel):
    """A result worth showing without being asked: a title, a sentence on what it means,
    and the query behind its chart. Kind and columns may be left to the result's shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    text: str
    sql: str
    kind: Literal["bar", "line", "scatter", "areas", "table"] | None = None
    x: str | None = None
    y: str | None = None
    color: str | None = None


class ShowcaseConfig(BaseModel):
    """What a published snapshot of the explorer may hold (`mlops export`).

    Every table the explorer's pages name goes, with every study and figure, unless a
    source's terms keep it home: a table here is never exported, and a page that reads it
    says it is not in the showcase. A table can also go only in part.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    withheld: dict[str, str] = {}  # layer.table -> why it stays home
    rows: dict[str, str] = {}  # layer.table -> the condition its exported rows meet
    credits: str = ""  # Markdown: whom the showcase's data comes from, as their terms ask

    @field_validator("withheld", "rows", mode="after")
    @classmethod
    def _tables_are_named_with_their_layer(cls, tables: dict[str, str]) -> dict[str, str]:
        unqualified = sorted(t for t in tables if not re.fullmatch(r"[a-z_]+\.\w+", t))
        if unqualified:
            raise ValueError(f"Name tables as layer.table: {unqualified}")
        return tables


class ExploreConfig(BaseModel):
    """The explorer app (`mlops explore`): its map and layers, the numbers over it, the
    tables a person can slice, the findings it opens with, and questions to start with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    intro: str = ""  # a paragraph under the title: what the app is about
    about: str = ""  # Markdown: the sources and their limits, in the domain's words
    view: MapView
    areas: AreasConfig | None = None
    layers: list[MapLayer] = []
    metrics: list[ExploreMetric] = []
    datasets: list[ExploreDataset] = []
    findings: list[ExploreFinding] = []
    examples: list[str] = []
    showcase: ShowcaseConfig = ShowcaseConfig()

    @property
    def queries(self) -> list[str]:
        """Every query and table name the pages hold: what a snapshot has to answer."""
        areas = [a.table for a in (self.areas, *(lay.areas for lay in self.layers)) if a]
        return [
            *areas,
            *(layer.sql for layer in self.layers),
            *(m.sql for m in self.metrics),
            *(m.trend for m in self.metrics if m.trend),
            *(f"{d.table} {d.where or ''}" for d in self.datasets),
            *(f.sql for f in self.findings),
        ]

    @model_validator(mode="after")
    def _an_areas_layer_has_areas(self) -> Self:
        drawn = [
            layer.name
            for layer in self.layers
            if layer.kind == "areas" and layer.areas is None and self.areas is None
        ]
        if drawn:
            raise ValueError(f"Layers {drawn} draw areas, but `explore.areas` names none")
        return self


class SqlGuard(BaseModel):
    """A table whose name says less than its rows hold, and the column a query of it must
    filter on - with the hint the agent's model gets when a query ignores it.

    Said in the data dictionary too; a small model reads past a sentence it was given,
    and a query that runs gives no error to repair from. The guard checks the query
    after it ran, and asks once."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str  # schema.table, as queries name it
    requires: str  # the column a query of the table should filter on
    hint: str


class AgentConfig(BaseModel):
    """What the agent's SQL is checked against, beyond what the database refuses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sql_guards: list[SqlGuard] = []
    # How many of the data dictionary's sections the SQL writer sees, chosen per question
    # by similarity (with the tables they name); unset, all of them.
    schema_sections: int | None = Field(default=None, ge=1)
    # How many queries the SQL writer writes and lets vote on the answer: the usual one,
    # and the rest sampled at `vote_temperature`, each with a seed of its own
    # (`text_to_sql.voted`). 1: no vote.
    sql_votes: int = Field(default=1, ge=1)
    vote_temperature: float = Field(default=0.7, gt=0, le=2)
    # Views the local agent is not shown (`schema.table`): each one it reads costs a small
    # model a little accuracy on the rest. They stay queryable, and the MCP server - whose
    # clients bring their own, larger models - still offers them.
    hidden_tables: list[str] = []
    # Models the local agent is not offered, for the same reason: the router reads every
    # model's description, and a longer list moves its choices (nine models sent two of ten
    # prediction questions to the tables that three had routed right). The API, the MCP
    # server and the explorer still serve them.
    hidden_models: list[str] = []

    def shown_models(self, models: list[ModelConfig]) -> list[ModelConfig]:
        """The domain's models the local agent may route to and ask."""
        return [model for model in models if model.name not in self.hidden_models]

    @field_validator("hidden_tables")
    @classmethod
    def _tables_are_qualified(cls, names: list[str]) -> list[str]:
        bad = [name for name in names if not re.fullmatch(r"\w+(\.\w+){1,2}", name)]
        if bad:
            raise ValueError(f"`agent.hidden_tables` names {bad}: write them as schema.table")
        return names


# The layers a domain's data directory holds, each a schema of its catalog.
DATA_LAYERS = ("clean", "features", "predictions", "analysis", "evaluations", "monitoring")


# Where a domain keeps its subdomains' data: `<domain>/subdomains/<name>/`, beside its own
# layers and never inside one, so no session of the domain's can read them.
SUBDOMAINS = "subdomains"


class ParentTables(BaseModel):
    """The domain a subdomain belongs to, and the tables of it the subdomain reads.

    Domains are tenants, isolated from each other: none reads another's data. A domain can
    hold subdomains - one per business, say - that share its code and read its tables,
    named one by one, read-only, in the newest partition it built. A subdomain reads
    nothing of its siblings, or of any other domain: the only tables it can name besides
    its own are its parent's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str
    tables: list[str] = Field(min_length=1)  # layer.table, as the parent names them

    @field_validator("tables")
    @classmethod
    def _tables_are_named_with_their_layer(cls, tables: list[str]) -> list[str]:
        bad = [t for t in tables if not re.fullmatch(rf"({'|'.join(DATA_LAYERS)})\.\w+", t)]
        if bad:
            raise ValueError(f"Name the parent's tables as layer.table: {bad}")
        return tables

    def qualified(self) -> list[str]:
        """The tables as the subdomain names them: `<parent>.<layer>.<table>`."""
        return [f"{self.domain}.{table}" for table in self.tables]


class DomainConfig(BaseModel):
    """The sections the core runs. A domain subclasses this to add its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sources: dict[str, SourceConfig]
    documents: list[DocumentConfig] = []  # the corpus; empty until a domain has one
    corpus: CorpusConfig | None = None  # required once there are documents
    models: list[ModelConfig] = Field(min_length=1)
    # Raw sources whose history is every download, not the latest: a page that shows only
    # the current month, a catalogue read week after week. Each download is checked on its
    # own and the frame is all of them, each row with its `ingested_at`; the domain decides
    # which reading of a row wins, and `prune` never touches them. File and API sources
    # alike, by name.
    accumulate: list[str] = []
    analysis: AnalysisConfig
    monitoring: MonitoringConfig
    schedule: ScheduleConfig | None = None  # unset, nothing runs until someone asks
    explore: ExploreConfig | None = None  # the explorer app's map; unset, it has none
    # Set, this is a subdomain of that domain, reading the tables of it listed
    # (`ParentTables`); unset, a domain of its own, which reads no other's data.
    parent: ParentTables | None = None
    agent: AgentConfig = AgentConfig()

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
    def _a_subdomain_is_not_its_own_parent(self) -> Self:
        if self.parent is not None and self.parent.domain == self.name:
            raise ValueError(f"{self.name} names itself as its parent")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"A domain's name is lower case, digits and _: '{self.name}'")
        return self

    @property
    def tenant(self) -> str:
        """How commands name it: `<domain>`, or `<parent>/<subdomain>`."""
        return f"{self.parent.domain}/{self.name}" if self.parent else self.name

    @property
    def home(self) -> Path:
        """Its data directory under the data root: `<domain>/`, or a subdomain's
        `<parent>/subdomains/<name>/`."""
        if self.parent is None:
            return Path(self.name)
        return Path(self.parent.domain, SUBDOMAINS, self.name)

    @property
    def parent_tables(self) -> set[str]:
        """Every table of its parent this subdomain may read, qualified; none for a domain."""
        return set(self.parent.qualified()) if self.parent else set()

    def readable(self, name: str) -> str:
        """`name`, if this domain may read it: one of its own clean tables (a bare name) or,
        for a subdomain, one of its parent's that it lists. Anything else is refused."""
        if name.count(".") == 2 and name not in self.parent_tables:
            whose = "lists in `parent`" if self.parent else "may read: it reads no other domain"
            raise ValueError(f"{name} is not a table {self.name} {whose}")
        return name

    @model_validator(mode="after")
    def _models_are_named_once(self) -> Self:
        names = [model.name for model in self.models]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Model names must be unique; repeated: {repeated}")
        unknown = sorted(set(self.agent.hidden_models) - set(names))
        if unknown:
            raise ValueError(f"`agent.hidden_models` names no model of the domain: {unknown}")
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
