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

from domains.coffee.config import DenueConfig
from mlops_core.data.api import ApiClient
from mlops_core.data.extract import RawArtifact, store_payload

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


# DENUE's staff-size strata, as its records name them (verified 2026-09-29: the counts per
# stratum of class 722515 in Cuauhtémoc match the records' `Estrato` band by band).
STRATA = {
    1: (0, 5, "0 a 5 personas"),
    2: (6, 10, "6 a 10 personas"),
    3: (11, 30, "11 a 30 personas"),
    4: (31, 50, "31 a 50 personas"),
    5: (51, 100, "51 a 100 personas"),
    6: (101, 250, "101 a 250 personas"),
    7: (251, None, "251 y más personas"),
}
COUNT_URL = "https://www.inegi.org.mx/app/api/denue/v1/consulta/Cuantificar"


def ingest_workplaces(
    client: ApiClient,
    config: DenueConfig,
    token: str,
    raw_dir: Path,
    now: datetime | None = None,
) -> RawArtifact:
    """Every establishment of every activity in each area, counted per staff-size stratum:
    `Cuantificar` with activity 0 answers a count for every sector, subsector, branch and
    class, without a record downloaded. Each answer is kept whole, in a canonical order."""
    workplaces = config.workplaces
    assert workplaces is not None  # the caller asks only when the config declares it
    answers = {}
    for area in workplaces.areas:
        for stratum in STRATA:
            url = f"{config.base_url}/Cuantificar/{ALL}/{area}/{stratum}/{token}"
            rows = client.get_json(url, cache_key=f"denue:workplaces:{area}:{stratum}")
            answers[f"{area}/{stratum}"] = sorted(rows, key=lambda row: row["AE"])
    payload = json.dumps(answers, ensure_ascii=False, sort_keys=True).encode("utf-8")
    logger.info("DENUE: every activity counted in %d areas, %d strata", len(workplaces.areas),
                len(STRATA))  # fmt: skip
    return store_payload(workplaces.name, workplaces.filename, payload, raw_dir, COUNT_URL, now)


def workplaces_frame(answers: dict[str, list[dict[str, str]]]) -> pl.DataFrame:
    """The stored counts as rows: area, stratum, activity code and how many there are."""
    rows = [
        (key.split("/")[0], int(key.split("/")[1]), row["AE"], row["Total"])
        for key, counted in answers.items()
        for row in counted
    ]
    return pl.DataFrame(
        rows, schema=["area", "stratum", "activity", "establishments"], orient="row"
    )


def to_frame(records: list[dict[str, str]]) -> pl.DataFrame:
    """The stored inventory as a frame, reshaped and not edited.

    Every DENUE field arrives as a string, including the coordinates; the contract in
    `schemas.py` is what types them and says which ones the pipeline depends on.
    """
    # infer_schema_length=None: a field that is empty in the first hundred records and
    # filled later must not be typed from the sample.
    return pl.DataFrame(records, infer_schema_length=None)
