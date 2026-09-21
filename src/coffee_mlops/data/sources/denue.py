"""DENUE (INEGI): every establishment of one activity class in one state.

The inventory comes from `BuscarAreaAct`, which pages 100 records at a time, and the
expected total from `Cuantificar`, which answers without downloading anything. Asking
for the count first turns a silent truncation — a page that came back short because the
service hiccuped — into a mismatch the caller can see.

⚠️ The token travels in the **URL path**, not a header. Nothing here logs a URL, and the
cache key is built from the query's meaning, so the credential never reaches a log line,
a manifest or a file name.
"""

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import polars as pl

from coffee_mlops.config import DenueConfig
from coffee_mlops.data.api import ApiClient
from coffee_mlops.data.extract import RawArtifact, store_payload

logger = logging.getLogger(__name__)

# What the documented endpoint is, for the manifest: the real request carries the token.
DOCUMENTED_URL = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaAct"
ALL = "0"  # DENUE's wildcard for "every municipality / locality / AGEB / block"


def count(client: ApiClient, config: DenueConfig, token: str) -> int:
    """How many establishments the service says exist, before downloading any."""
    url = f"{config.base_url}/Cuantificar/{config.activity_class}/{config.entity}/0/{token}"
    payload = client.get_json(url, cache_key=f"denue:count:{config.activity_class}:{config.entity}")
    return int(payload[0]["Total"])


def establishments(client: ApiClient, config: DenueConfig, token: str) -> Iterator[dict[str, str]]:
    """Every record of the class, page by page, stopping on the first short page."""
    start = 1
    while True:
        end = start + config.page_size - 1
        url = (
            f"{config.base_url}/BuscarAreaAct/{config.entity}/{ALL}/{ALL}/{ALL}/{ALL}"
            f"/0/0/0/{config.activity_class}/0/{start}/{end}/0/{token}"
        )
        page = client.get_json(
            url, cache_key=f"denue:{config.activity_class}:{config.entity}:{start}-{end}"
        )
        yield from page
        if len(page) < config.page_size:
            return
        start = end + 1


def ingest_establishments(
    client: ApiClient,
    config: DenueConfig,
    token: str,
    raw_dir: Path,
    now: datetime | None = None,
) -> RawArtifact:
    """Collect every page into one raw JSON file, checked against the service's count."""
    expected = count(client, config, token)
    records = list(establishments(client, config, token))
    if len(records) != expected:
        # Not fatal: the count and the inventory are two calls and the register moves.
        # Loud, though, because a short page looks exactly like a complete one.
        logger.warning(
            "DENUE returned %d records but reported %d for class %s in %s",
            len(records),
            expected,
            config.activity_class,
            config.entity,
        )
    logger.info("DENUE: %d establishments of class %s", len(records), config.activity_class)
    # Canonical form before storing: paging a live register hands the same rows back in a
    # different order from one run to the next, so without this every run would look like
    # new data and the raw layer would fill with identical partitions.
    records.sort(key=lambda record: record["Id"])
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return store_payload(config.name, config.filename, payload, raw_dir, DOCUMENTED_URL, now)


def to_frame(records: list[dict[str, str]]) -> pl.DataFrame:
    """The stored inventory as a frame, reshaped and not edited.

    Every DENUE field arrives as a string, including the coordinates; the contract in
    `schemas.py` is what types them and says which ones the pipeline depends on.
    """
    # infer_schema_length=None: a field that is empty in the first hundred records and
    # filled later must not be typed from the sample.
    return pl.DataFrame(records, infer_schema_length=None)
