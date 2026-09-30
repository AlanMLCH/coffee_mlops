"""Quality gate between raw and clean: read the latest ingestion of each source and
validate it against the contract the domain declares for it. Any violation stops the
pipeline.

Several kinds of source arrive here and each is *read* differently while being
*checked* the same way: a table (CSV, possibly inside a ZIP, or a workbook's sheet), a
map layer (read in place out of the archive and reprojected), a file only the domain can
read (a PDF laid out as a table), and an API's stored JSON (flattened by the domain
module that knows that service's shape). A source that has never been ingested because
its credential is missing is reported and skipped, not raised: the rest of the pipeline
still has work to do.

A source whose history is its downloads (named in the config's `accumulate`, file or API
alike) is read whole: every ingestion, each checked against the contract on its own,
stacked with the time of its download. One such download can also be checked alone
(`validate_read`): the orchestrator does it read by read, a partition each.
"""

import json
import logging
import re
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandera.polars as pa
import polars as pl

from mlops_core.adapter import DomainAdapter, FileReader, JsonReader
from mlops_core.config import SourceConfig
from mlops_core.contracts import check_contract
from mlops_core.data.documents import DOCUMENT_PARTS, read_document
from mlops_core.data.extract import RawArtifact, ingestions, latest_ingestion
from mlops_core.data.geo import read_areas

logger = logging.getLogger(__name__)

INGESTED_AT = "ingested_at"  # the column an accumulated source's rows gain


@dataclass(frozen=True)
class ValidatedSource:
    artifact: RawArtifact  # where the frame came from, for lineage: the latest download
    frame: pl.DataFrame
    reads: int = 1  # how many downloads the frame stacks

    @property
    def lineage(self) -> str:
        """The raw partition the frame came from, or the range an accumulated one spans."""
        name = self.artifact.partition.name
        return name if self.reads == 1 else f"{name} and {self.reads - 1} earlier"


def read_raw(artifact: RawArtifact, source: SourceConfig) -> pl.DataFrame:
    """Read a raw file with every column as text; the schema does the typing.

    The raw file stays in its own encoding on disk (the raw layer never transforms);
    it is decoded here, on read, because polars parses UTF-8 only.
    """
    if source.sheet is not None:
        return read_sheet(artifact.path, source)
    if source.member is None:
        data = artifact.path.read_bytes()
    else:
        with zipfile.ZipFile(artifact.path) as archive:
            data = archive.read(source.member)
    if source.encoding.replace("-", "").lower() != "utf8":
        data = data.decode(source.encoding).encode("utf-8")
    frame = pl.read_csv(data, infer_schema_length=0, null_values=source.null_values or None)
    frame = frame.rename({old: new for old, new in source.renamed.items() if old in frame.columns})
    if source.thousands is not None:
        frame = frame.with_columns(_without_thousands(frame, source.thousands))
    return frame


def _without_thousands(frame: pl.DataFrame, separator: str) -> list[pl.Expr]:
    """Each text column with the separator taken out of the cells that are a number
    written with it ("3,350.00"), and nothing else touched."""
    mark = re.escape(separator)
    number = rf"^-?\d{{1,3}}({mark}\d{{3}})+(\.\d+)?$"
    return [
        pl.when(pl.col(c).str.contains(number))
        .then(pl.col(c).str.replace_all(separator, "", literal=True))
        .otherwise(pl.col(c))
        .alias(c)
        for c in frame.columns
        if frame.schema[c] == pl.String
    ]


def read_sheet(path: Path, source: SourceConfig) -> pl.DataFrame:
    """A workbook's sheet as text columns: named by its header row, the rows above it
    (titles, notes) and the `skip_rows` under it (units) left out. A column without a
    name is `column_<n>`, counted from 1 as a spreadsheet counts."""
    cells = pl.read_excel(path, sheet_name=source.sheet, has_header=False, infer_schema_length=0)
    names = [
        str(name).strip() if name is not None else f"column_{n}"
        for n, name in enumerate(cells.row(source.header_row), start=1)
    ]
    body = cells.slice(source.header_row + 1 + source.skip_rows)
    return body.rename(dict(zip(body.columns, names, strict=True)))


def read_layer(artifact: RawArtifact, source: SourceConfig) -> pl.DataFrame:
    """Read a map layer out of the archive it was downloaded in, without unpacking it."""
    if source.member is None or source.spatial is None:  # pragma: no cover - config model
        raise ValueError(f"{source.filename} has no spatial layer configured")
    return read_areas(artifact.path, source.member, source.spatial)


def validate_raw(adapter: DomainAdapter, raw_dir: Path) -> dict[str, ValidatedSource]:
    """Return each source validated and typed, or raise `SchemaErrors` listing every failure."""
    config = adapter.config
    contracts = adapter.raw_contracts()
    readers = adapter.file_readers()
    json_readers = adapter.json_readers()
    unknown = sorted(set(readers) - set(config.sources))
    if unknown:
        raise ValueError(f"Readers for files the config does not download: {unknown}")
    nameless = sorted(set(config.accumulate) - set(config.sources) - set(json_readers))
    if nameless:
        raise ValueError(f"`accumulate` names sources that do not exist: {nameless}")
    history = set(config.accumulate)
    validated = {}
    for name in config.sources:
        if latest_ingestion(raw_dir, name) is None:
            raise FileNotFoundError(
                f"No raw ingestion for '{name}' in {raw_dir}; run extract first"
            )
        read = _reader(adapter, name, readers, json_readers)
        yearly = config.sources[name].years is not None
        validated[name] = _validated(
            name, raw_dir, contracts[name], name in history, read, yearly=yearly
        )

    for name in json_readers:
        if latest_ingestion(raw_dir, name) is None:
            # Not an error: a source whose credential is missing is skipped at extract,
            # and everything that does not depend on it still builds.
            logger.info("%s has never been ingested; skipping its contract", name)
            continue
        read = _reader(adapter, name, readers, json_readers)
        validated[name] = _validated(name, raw_dir, contracts[name], name in history, read)

    for document in adapter.config.documents:
        artifact = latest_ingestion(raw_dir, document.name)
        if artifact is None:
            # A document nobody handed over yet: the corpus is short, not broken.
            logger.info("%s has never been ingested; skipping its contract", document.name)
            continue
        validated[document.name] = _checked(
            DOCUMENT_PARTS, artifact, read_document(artifact, document)
        )
    return validated


def validate_read(adapter: DomainAdapter, name: str, artifact: RawArtifact) -> ValidatedSource:
    """One download of a source, checked on its own against the source's contract."""
    read = _reader(adapter, name, adapter.file_readers(), adapter.json_readers())
    return _checked(adapter.raw_contracts()[name], artifact, read(artifact))


def _reader(
    adapter: DomainAdapter,
    name: str,
    readers: Mapping[str, FileReader],
    json_readers: Mapping[str, JsonReader],
) -> Callable[[RawArtifact], pl.DataFrame]:
    """How a download of the source is read: a file the config describes, or the JSON an
    API answered, flattened by the domain."""
    source = adapter.config.sources.get(name)
    if source is not None:
        return partial(_read, source=source, readers=readers)
    return partial(_read_json, reader=json_readers[name])


def _validated(
    name: str,
    raw_dir: Path,
    contract: pa.DataFrameSchema,
    accumulate: bool,
    read: Callable[[RawArtifact], pl.DataFrame],
    yearly: bool = False,
) -> ValidatedSource:
    """The latest download, checked; or, for a source whose history is its downloads,
    every one of them checked on its own and stacked with the time it was read; or, for a
    source of one file a year, each year's latest download, checked and stacked."""
    downloads = ingestions(raw_dir, name)
    if yearly:
        by_year = list({artifact.manifest.url: artifact for artifact in downloads}.values())
        frames = [_checked(contract, artifact, read(artifact)).frame for artifact in by_year]
        return ValidatedSource(downloads[-1], pl.concat(frames), len(by_year))
    if not accumulate:
        return _checked(contract, downloads[-1], read(downloads[-1]))
    frames = [
        _checked(contract, artifact, read(artifact)).frame.with_columns(
            pl.lit(artifact.manifest.ingested_at).alias(INGESTED_AT)
        )
        for artifact in downloads
    ]
    return ValidatedSource(downloads[-1], pl.concat(frames), len(downloads))


def _read_json(artifact: RawArtifact, reader: JsonReader) -> pl.DataFrame:
    return reader(json.loads(artifact.path.read_text(encoding="utf-8")))


def _read(
    artifact: RawArtifact, source: SourceConfig, readers: Mapping[str, FileReader]
) -> pl.DataFrame:
    reader = readers.get(artifact.manifest.source)
    if reader is not None:
        return reader(artifact.path)
    return read_layer(artifact, source) if source.spatial else read_raw(artifact, source)


def _checked(
    contract: pa.DataFrameSchema, artifact: RawArtifact, frame: pl.DataFrame
) -> ValidatedSource:
    checked = check_contract(contract, frame)
    logger.info(
        "%s valid: %d rows (%s)", artifact.manifest.source, checked.height, artifact.partition.name
    )
    return ValidatedSource(artifact, checked)
