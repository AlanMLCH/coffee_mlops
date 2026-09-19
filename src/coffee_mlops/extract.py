"""Raw layer: download each source byte-for-byte, with an ingestion manifest.

Layout per source::

    <raw_dir>/<source>/ingested_at=<UTC timestamp>/<filename>
                                                  /manifest.json

Files are never transformed here. The manifest is written last, so a partition
without one is an interrupted download and is ignored. A download whose sha256
matches the latest ingestion is not stored again.
"""

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from coffee_mlops.config import DomainConfig, SourceConfig

logger = logging.getLogger(__name__)

USER_AGENT = "coffee-mlops/0.1 (+https://github.com/AlanMLCH/coffee_mlops)"
MANIFEST_NAME = "manifest.json"
PARTITION_PREFIX = "ingested_at="
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class Manifest(BaseModel):
    source: str
    # The configured URL, never the final one: redirects may land on signed URLs.
    url: str
    filename: str
    sha256: str
    size_bytes: int
    ingested_at: datetime
    last_modified: str | None = None


@dataclass(frozen=True)
class RawArtifact:
    partition: Path
    manifest: Manifest

    @property
    def path(self) -> Path:
        return self.partition / self.manifest.filename


@contextmanager
def http_client(transport: httpx.BaseTransport | None = None) -> Iterator[httpx.Client]:
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
        transport=transport,
    ) as client:
        yield client


def latest_ingestion(raw_dir: Path, source: str) -> RawArtifact | None:
    partitions = sorted(
        p for p in (raw_dir / source).glob(f"{PARTITION_PREFIX}*") if (p / MANIFEST_NAME).is_file()
    )
    if not partitions:
        return None
    manifest = Manifest.model_validate_json((partitions[-1] / MANIFEST_NAME).read_text())
    return RawArtifact(partitions[-1], manifest)


def ingest(
    name: str,
    source: SourceConfig,
    raw_dir: Path,
    client: httpx.Client,
    now: datetime | None = None,
) -> RawArtifact:
    source_dir = raw_dir / name
    source_dir.mkdir(parents=True, exist_ok=True)
    part_file = source_dir / f".{source.filename}.part"

    sha256, size, last_modified = _download(client, str(source.url), part_file)

    previous = latest_ingestion(raw_dir, name)
    if previous is not None and previous.manifest.sha256 == sha256:
        part_file.unlink()
        logger.info("%s unchanged since %s", name, previous.manifest.ingested_at)
        return previous

    ingested_at = now or datetime.now(UTC)
    partition = source_dir / f"{PARTITION_PREFIX}{ingested_at.strftime(TIMESTAMP_FORMAT)}"
    partition.mkdir()
    part_file.replace(partition / source.filename)
    manifest = Manifest(
        source=name,
        url=str(source.url),
        filename=source.filename,
        sha256=sha256,
        size_bytes=size,
        ingested_at=ingested_at,
        last_modified=last_modified,
    )
    (partition / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))
    logger.info("%s ingested: %s (%d bytes)", name, partition, size)
    return RawArtifact(partition, manifest)


def extract_all(
    config: DomainConfig, raw_dir: Path, client: httpx.Client
) -> dict[str, RawArtifact]:
    return {name: ingest(name, source, raw_dir, client) for name, source in config.sources.items()}


def _download(client: httpx.Client, url: str, target: Path) -> tuple[str, int, str | None]:
    """Stream `url` into `target`; return (sha256, size, Last-Modified header)."""
    digest = hashlib.sha256()
    size = 0
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            last_modified = response.headers.get("last-modified")
        if size == 0:
            raise ValueError(f"Empty response body from {url}")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size, last_modified
