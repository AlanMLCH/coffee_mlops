"""Quality gate between raw and clean: read the latest ingestion of each source and
validate it against the contract the domain declares for it. Any violation stops the
pipeline.

Three kinds of source arrive here and each is *read* differently while being *checked*
the same way: a table (CSV, possibly inside a ZIP), a map layer (read in place out of
the archive and reprojected), and an API's stored JSON (flattened by the domain module
that knows that service's shape). A source that has never been ingested because its
credential is missing is reported and skipped, not raised: the rest of the pipeline
still has work to do.
"""

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandera.polars as pa
import polars as pl

from mlops_core.adapter import DomainAdapter
from mlops_core.config import SourceConfig
from mlops_core.contracts import check_contract
from mlops_core.data.extract import RawArtifact, latest_ingestion
from mlops_core.data.geo import read_areas

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatedSource:
    artifact: RawArtifact  # where the frame came from, for lineage
    frame: pl.DataFrame


def read_raw(artifact: RawArtifact, source: SourceConfig) -> pl.DataFrame:
    """Read a raw file with every column as text; the schema does the typing."""
    if source.member is None:
        data = artifact.path.read_bytes()
    else:
        with zipfile.ZipFile(artifact.path) as archive:
            data = archive.read(source.member)
    return pl.read_csv(data, infer_schema_length=0, null_values=source.null_values or None)


def read_layer(artifact: RawArtifact, source: SourceConfig) -> pl.DataFrame:
    """Read a map layer out of the archive it was downloaded in, without unpacking it."""
    if source.member is None or source.spatial is None:  # pragma: no cover - config model
        raise ValueError(f"{source.filename} has no spatial layer configured")
    return read_areas(artifact.path, source.member, source.spatial)


def validate_raw(adapter: DomainAdapter, raw_dir: Path) -> dict[str, ValidatedSource]:
    """Return each source validated and typed, or raise `SchemaErrors` listing every failure."""
    contracts = adapter.raw_contracts()
    validated = {}
    for name, source in adapter.config.sources.items():
        artifact = latest_ingestion(raw_dir, name)
        if artifact is None:
            raise FileNotFoundError(
                f"No raw ingestion for '{name}' in {raw_dir}; run extract first"
            )
        frame = read_layer(artifact, source) if source.spatial else read_raw(artifact, source)
        validated[name] = _checked(contracts[name], artifact, frame)

    for name, read_json in adapter.json_readers().items():
        artifact = latest_ingestion(raw_dir, name)
        if artifact is None:
            # Not an error: a source whose credential is missing is skipped at extract,
            # and everything that does not depend on it still builds.
            logger.info("%s has never been ingested; skipping its contract", name)
            continue
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
        validated[name] = _checked(contracts[name], artifact, read_json(payload))
    return validated


def _checked(
    contract: pa.DataFrameSchema, artifact: RawArtifact, frame: pl.DataFrame
) -> ValidatedSource:
    checked = check_contract(contract, frame)
    logger.info(
        "%s valid: %d rows (%s)", artifact.manifest.source, checked.height, artifact.partition.name
    )
    return ValidatedSource(artifact, checked)
