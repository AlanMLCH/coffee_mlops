"""USDA FAS Open Data: the PSD coffee balance, one market year per request.

The same data as the stage 1 ZIP, reached through a different archetype: an API that
authenticates with a key in a **header** (DENUE's token rides in the URL path) and
answers by year. Verified on 2026-09-21 against the ZIP: 87,704 rows each, the same
keys, and not one value different. That equivalence is what lets this source be checked
against the file on every build instead of trusted on faith.

What the service is like, and what that asks of this module:

- **It answers with ids.** A row says `attributeId: 29`, not "Arabica Production", so the
  four lookup tables (commodities, attributes, units, countries) are stored with the
  rows. Reading the data back must not depend on asking the service again.
- **It pads.** Unit descriptions come fixed-width (`"(1000 60 KG BAGS)   "`); the file
  does not pad them. `to_frame` strips them, or the shared contract would refuse them.
- **A year past the last one is an empty list, not an error.** So the walk asks up to
  the year after the current one - where PSD puts its forecast - and an empty year costs
  one request and adds nothing.
- **1,000 requests per hour per key** (`x-ratelimit-limit`). A full pull is ~70.
"""

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from coffee_mlops.config import FasConfig
from coffee_mlops.data.api import ApiClient
from coffee_mlops.data.extract import RawArtifact, store_payload

logger = logging.getLogger(__name__)

# api.data.gov's convention. In a header, never `?api_key=`: a header never ends up in a
# URL, and so never in a log line, a cache key or a manifest.
KEY_HEADER = "X-Api-Key"
# Lookup name -> endpoint, and the field each one is sorted by for a canonical form.
LOOKUPS = {
    "commodities": ("commodities", "commodityCode"),
    "attributes": ("commodityAttributes", "attributeId"),
    "units": ("unitsOfMeasure", "unitId"),
    "countries": ("countries", "countryCode"),
}


def fetch_rows(
    client: ApiClient, config: FasConfig, key: str, last_year: int
) -> Iterator[dict[str, Any]]:
    """Every row of the commodity, market year by market year, oldest first."""
    for year in range(config.first_year, last_year + 1):
        yield from _get(
            client, config, key, f"commodity/{config.commodity_code}/country/all/year/{year}"
        )


def ingest_balance(
    client: ApiClient, config: FasConfig, key: str, raw_dir: Path, now: datetime | None = None
) -> RawArtifact:
    """Store the rows and the lookups that decode them as one raw JSON document."""
    moment = now or datetime.now(UTC)
    rows = list(fetch_rows(client, config, key, last_year=moment.year + 1))
    if not rows:
        # An empty pull stored as a raw partition would read as "the market vanished".
        raise RuntimeError(f"FAS returned no rows for commodity {config.commodity_code}")
    document: dict[str, Any] = {
        name: sorted(_get(client, config, key, endpoint), key=lambda item: item[field])
        for name, (endpoint, field) in LOOKUPS.items()
    }
    # Same reason as the other APIs: without a canonical order, an identical answer
    # hashes differently and the raw layer fills with copies of itself.
    document["rows"] = sorted(
        rows, key=lambda r: (r["marketYear"], r["countryCode"], r["attributeId"])
    )
    logger.info(
        "FAS: %d rows for commodity %s, market years %s-%s",
        len(rows),
        config.commodity_code,
        document["rows"][0]["marketYear"],
        document["rows"][-1]["marketYear"],
    )
    body = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return store_payload(config.name, config.filename, body, raw_dir, config.base_url, now)


def to_frame(document: dict[str, Any]) -> pl.DataFrame:
    """The stored rows, decoded, in exactly the columns of the PSD file.

    Same columns, same contract: the file and the API are two ways to reach one table,
    and holding them to one schema is what makes them comparable. An id missing from its
    lookup decodes to null and is refused by the contract, rather than guessed.
    """
    names = {
        "commodities": {c["commodityCode"]: c["commodityName"] for c in document["commodities"]},
        "attributes": {a["attributeId"]: a["attributeName"] for a in document["attributes"]},
        "units": {u["unitId"]: u["unitDescription"].strip() for u in document["units"]},
        "countries": {c["countryCode"]: c["countryName"] for c in document["countries"]},
    }
    return pl.DataFrame(
        [
            {
                "Commodity_Code": row["commodityCode"],
                "Commodity_Description": names["commodities"].get(row["commodityCode"]),
                "Country_Code": row["countryCode"],
                "Country_Name": names["countries"].get(row["countryCode"]),
                "Market_Year": row["marketYear"],
                "Calendar_Year": row["calendarYear"],
                "Month": row["month"],
                "Attribute_ID": row["attributeId"],
                "Attribute_Description": names["attributes"].get(row["attributeId"]),
                "Unit_ID": row["unitId"],
                "Unit_Description": names["units"].get(row["unitId"]),
                "Value": row["value"],
            }
            for row in document["rows"]
        ],
        infer_schema_length=None,
    )


def _get(client: ApiClient, config: FasConfig, key: str, endpoint: str) -> list[dict[str, Any]]:
    # The key is in a header, so the endpoint alone identifies the request: safe to cache
    # on, and it changes whenever the request does.
    payload: list[dict[str, Any]] = client.get_json(
        f"{config.base_url}/{endpoint}", cache_key=f"fas:{endpoint}", headers={KEY_HEADER: key}
    )
    return payload
