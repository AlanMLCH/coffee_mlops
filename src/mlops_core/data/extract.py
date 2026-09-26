"""Raw layer: download each source byte-for-byte, with an ingestion manifest.

Layout per source::

    <raw_dir>/<source>/ingested_at=<UTC timestamp>/<filename>
                                                  /manifest.json

Files are never transformed here. The manifest is written last, so a partition
without one is an interrupted download and is ignored. A download whose sha256
matches the latest ingestion is not stored again.

A connection the server cuts is tried again, up to three times: one host this project
reads resets connections now and then - twice in a row on its first real run - and
answers a later request. An HTTP error is not retried: a 404 is an answer.
"""

import hashlib
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import distributions
from pathlib import Path
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel

from mlops_core.config import DomainConfig, SourceConfig
from mlops_core.data.api import silence_request_urls
from mlops_core.storage import MANIFEST_NAME, latest_partition, new_partition

logger = logging.getLogger(__name__)

DOWNLOAD_ATTEMPTS = 4
_HREF = re.compile(r"""href=["']([^"']+)["']""")


class Manifest(BaseModel):
    source: str
    # The configured URL, never the final one: redirects may land on signed URLs. A file
    # found by a link on a page records the link, which names the release.
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
        headers={"User-Agent": user_agent()},
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True,
        transport=transport,
    ) as client:
        yield client


@cache
def user_agent() -> str:
    """`<distribution>/<version> (+<repository>)`, from the installed package's metadata.

    Identifiable on purpose - an upstream that sees trouble can say who to contact - and
    read from packaging rather than written here, because it names the project that
    ships this core, which the core itself does not know. The distribution is found by
    the command it installs: an editable install records its entry points, not its
    packages.
    """
    package = __name__.split(".")[0]
    for dist in distributions():
        if any(point.value.startswith(f"{package}.") for point in dist.entry_points):
            links = dict(url.split(", ", 1) for url in dist.metadata.get_all("Project-URL") or [])
            home = f" (+{links['Repository']})" if "Repository" in links else ""
            return f"{dist.metadata['Name']}/{dist.version}{home}"
    return package  # running from a bare source tree: still says what is calling


def latest_ingestion(raw_dir: Path, source: str) -> RawArtifact | None:
    partition = latest_partition(raw_dir / source)
    if partition is None:
        return None
    return _artifact(partition)


def ingestions(raw_dir: Path, source: str) -> list[RawArtifact]:
    """Every complete ingestion of a source, oldest first: the history of a source
    whose each download is a window."""
    complete = sorted(p for p in (raw_dir / source).glob("*=*") if (p / MANIFEST_NAME).is_file())
    return [_artifact(partition) for partition in complete]


def ingest(
    name: str,
    source: SourceConfig,
    raw_dir: Path,
    client: httpx.Client,
    now: datetime | None = None,
) -> RawArtifact:
    url = (
        str(source.url) if source.link is None else find_link(client, str(source.url), source.link)
    )
    return ingest_file(name, url, source.filename, raw_dir, client, now)


def find_link(client: httpx.Client, page: str, pattern: str) -> str:
    """The first link on `page` that `pattern` matches, as an absolute URL."""
    response = client.get(page)
    response.raise_for_status()
    for href in _HREF.findall(response.text):
        if re.search(pattern, href):
            return urljoin(page, str(href))
    raise LookupError(f"No link on {page} matches {pattern!r}: look at the page and update `link`")


def ingest_file(
    name: str,
    url: str,
    filename: str,
    raw_dir: Path,
    client: httpx.Client,
    now: datetime | None = None,
) -> RawArtifact:
    """Stream a file into a raw partition: a table, a map layer, a document, any bytes."""
    source_dir = raw_dir / name
    source_dir.mkdir(parents=True, exist_ok=True)
    part_file = source_dir / f".{filename}.part"

    sha256, size, last_modified = _download(client, url, part_file)
    return _store(name, filename, part_file, raw_dir, url, sha256, size, last_modified, now)


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
    """Stream `url` into `target`; return (sha256, size, Last-Modified header). A dropped
    connection is tried again, waiting 1, 2 and then 4 s."""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            return _download_once(client, url, target)
        except httpx.TransportError as dropped:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            logger.warning("%s: %s; trying again", url, dropped)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


def _download_once(client: httpx.Client, url: str, target: Path) -> tuple[str, int, str | None]:
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


def _artifact(partition: Path) -> RawArtifact:
    return RawArtifact(
        partition, Manifest.model_validate_json((partition / MANIFEST_NAME).read_text())
    )
