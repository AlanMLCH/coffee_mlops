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
from coffee_mlops.data.api import silence_request_urls
from coffee_mlops.storage import MANIFEST_NAME, latest_partition, new_partition

logger = logging.getLogger(__name__)

USER_AGENT = "coffee-mlops/0.1 (+https://github.com/AlanMLCH/coffee_mlops)"


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
    silence_request_urls()  # a redirect can land on a signed URL; never log one
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
        transport=transport,
    ) as client:
        yield client


def latest_ingestion(raw_dir: Path, source: str) -> RawArtifact | None:
    partition = latest_partition(raw_dir / source)
    if partition is None:
        return None
    manifest = Manifest.model_validate_json((partition / MANIFEST_NAME).read_text())
    return RawArtifact(partition, manifest)


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
    return _store(
        name, source.filename, part_file, raw_dir, str(source.url), sha256, size, last_modified, now
    )


def store_payload(
    name: str,
    filename: str,
    payload: bytes,
    raw_dir: Path,
    source_url: str,
    now: datetime | None = None,
) -> RawArtifact:
    """Write bytes that were assembled rather than downloaded (an API's pages, joined)
    as a raw ingestion, with the same manifest, de-duplication and atomicity.

    `source_url` is the documented endpoint, never a URL carrying a credential.
    """
    source_dir = raw_dir / name
    source_dir.mkdir(parents=True, exist_ok=True)
    part_file = source_dir / f".{filename}.part"
    part_file.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    return _store(name, filename, part_file, raw_dir, source_url, sha256, len(payload), None, now)


def _store(
    name: str,
    filename: str,
    part_file: Path,
    raw_dir: Path,
    source_url: str,
    sha256: str,
    size: int,
    last_modified: str | None,
    now: datetime | None,
) -> RawArtifact:
    """The one place a raw partition is created, whatever produced the bytes."""
    previous = latest_ingestion(raw_dir, name)
    if previous is not None and previous.manifest.sha256 == sha256:
        part_file.unlink()
        logger.info("%s unchanged since %s", name, previous.manifest.ingested_at)
        return previous

    ingested_at = now or datetime.now(UTC)
    partition = new_partition(raw_dir / name, "ingested_at", ingested_at)
    part_file.replace(partition / filename)
    manifest = Manifest(
        source=name,
        url=source_url,
        filename=filename,
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
