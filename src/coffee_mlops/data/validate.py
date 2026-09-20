"""Quality gate between raw and clean: read the latest ingestion of each source and
validate it against its Pandera contract. Any violation stops the pipeline."""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from coffee_mlops.config import DomainConfig, SourceConfig
from coffee_mlops.contracts import check_contract
from coffee_mlops.data.extract import RawArtifact, latest_ingestion
from coffee_mlops.data.schemas import RAW_SCHEMAS

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


def validate_raw(config: DomainConfig, raw_dir: Path) -> dict[str, ValidatedSource]:
    """Return each source validated and typed, or raise `SchemaErrors` listing every failure."""
    validated = {}
    for name, source in config.sources.items():
        artifact = latest_ingestion(raw_dir, name)
        if artifact is None:
            raise FileNotFoundError(
                f"No raw ingestion for '{name}' in {raw_dir}; run extract first"
            )
        frame = check_contract(RAW_SCHEMAS[name], read_raw(artifact, source))
        validated[name] = ValidatedSource(artifact, frame)
        logger.info("%s valid: %d rows (%s)", name, frame.height, artifact.partition.name)
    return validated
