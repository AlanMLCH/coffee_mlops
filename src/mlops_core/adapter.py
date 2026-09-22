"""The contract between the core and a domain.

The core runs the whole cycle - extract, validate, clean, features, train, predict,
serve, analyse - without knowing what an item is. A domain answers only the questions
nobody else can: what its sources promise, how its raw tables become clean ones, and
what context an item is allowed to see. Whatever is data rather than code (URLs, column
names, the model's features) lives in the domain's YAML, not here.

The rule this module exists to enforce: a new domain is a new package under `domains/`
and nothing in `mlops_core` changes. If a domain needs the core edited, the contract was
wrong, and it gets fixed here rather than patched around.

A domain package exposes one function, `adapter()`, returning an object that satisfies
`DomainAdapter`. It is found by name, so the core never imports a domain by hand.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, SecretStr

from mlops_core.config import DomainConfig

if TYPE_CHECKING:  # the API image has neither httpx nor matplotlib installed
    import httpx
    from matplotlib.figure import Figure

    from mlops_core.data.extract import RawArtifact

DOMAINS_PACKAGE = "domains"

# An API source's stored JSON -> one frame, reshaped and not edited.
JsonReader = Callable[[Any], pl.DataFrame]


@dataclass(frozen=True)
class CleanTable:
    """A clean table and the raw sources it was built from, for its manifest."""

    frame: pl.DataFrame
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class ApiExtraction:
    """What the API sources brought back, and what was skipped and why.

    A missing credential is reported, not raised: a fresh clone has no `.env`, and
    everything that does not depend on that credential must still build. Skipping has
    to be loud, though, or an empty layer looks like an upstream with no rows.
    """

    artifacts: dict[str, RawArtifact] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # source name -> why


class ItemRequest(Protocol):
    """What the prediction API accepts: what a caller knows before the item is measured."""

    def to_item(self) -> dict[str, Any]:
        """The request as one row of the item table, time column included."""
        ...


class DomainAdapter(Protocol):
    """What a domain must answer for the core to run its whole cycle."""

    @property
    def config(self) -> DomainConfig:
        """The domain's YAML, validated against the domain's own config model."""
        ...

    def credentials(self) -> Mapping[str, SecretStr | None]:
        """Label -> secret, for `mlops secrets`. Empty when every source is open."""
        ...

    def extract(
        self, data_dir: Path, client: httpx.Client, now: datetime | None = None
    ) -> ApiExtraction:
        """Pull the API sources into `<data_dir>/raw`. The file sources in the YAML are
        downloaded by the core; only what needs code (paging, a query, a key) is here."""
        ...

    def raw_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        """One Pandera contract per raw source, file and API alike."""
        ...

    def json_readers(self) -> Mapping[str, JsonReader]:
        """How each API source's stored JSON becomes a frame for its contract."""
        ...

    def clean(self, raw: Mapping[str, pl.DataFrame]) -> Mapping[str, CleanTable]:
        """Validated raw frames -> the domain's canonical tables: the items and every
        context table that describes their world."""
        ...

    def clean_contracts(self) -> Mapping[str, pa.DataFrameSchema]:
        """One strict contract per clean table: the domain's promise to every reader."""
        ...

    def context_tables(self) -> tuple[str, ...]:
        """The clean tables `enrich` reads. Serving loads exactly these."""
        ...

    def enrich(self, items: pl.DataFrame, context: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
        """Add to each item the context it may see - nothing dated after its time column.

        One function for the batch feature table and for every online request, so the
        two can never compute a feature differently.
        """
        ...

    def request_model(self) -> type[BaseModel]:
        """The prediction API's request body. It must implement `ItemRequest`."""
        ...

    def studies(self, clean: Mapping[str, pl.DataFrame]) -> Mapping[str, pl.DataFrame]:
        """Analyses only this domain asks for, on top of the ones every domain gets."""
        ...

    def figures(self, tables: Mapping[str, pl.DataFrame]) -> Mapping[str, Figure]:
        """Figures for the domain's own studies. A study that came out empty is skipped."""
        ...


def available_domains() -> list[str]:
    """Every domain package installed under `domains/`."""
    package = importlib.import_module(DOMAINS_PACKAGE)
    return sorted(module.name for module in pkgutil.iter_modules(package.__path__))


def load_adapter(domain: str | None = None) -> DomainAdapter:
    """The named domain's adapter; with no name, the only one installed.

    Picking the lone domain is a convenience, not a guess: with two installed, a
    command that does not say which one refuses to run.
    """
    if domain is None:
        installed = available_domains()
        if len(installed) != 1:
            raise ValueError(f"Name a domain (--domain or MLOPS_DOMAIN); installed: {installed}")
        domain = installed[0]
    try:
        module = importlib.import_module(f"{DOMAINS_PACKAGE}.{domain}")
    except ModuleNotFoundError as missing:
        if missing.name != f"{DOMAINS_PACKAGE}.{domain}":
            raise  # the domain exists but one of its own imports failed: say that instead
        raise ValueError(f"No domain '{domain}'; installed: {available_domains()}") from missing
    adapter: DomainAdapter = module.adapter()
    return adapter
