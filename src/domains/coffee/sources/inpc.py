"""INEGI's national consumer price index (INPC), from its indicators API.

What turns a peso of 2003 into a peso of today: SIAP's rural prices since 2003, the green
coffee price in pesos since 1993 and PROFECO's shelves since 2024 are all nominal. One
request brings the whole monthly series. Verified 2026-10-01 with the project's client:

- ⚠️ The token travels in the **URL path**, like DENUE's. Nothing here logs a URL; the
  cache key and the manifest carry the documented endpoint, never the token.
- The bank is `BIE-BISE`: asked of `BIE` or `BISE` alone the index answers 400 "No se
  encontraron resultados", and so does the area `0700` the documentation shows; the
  country is `00`.
- The answer: `Series[0]` with `INDICADOR`, `FREQ`, `UNIT`, `LASTUPDATE` and
  `OBSERVATIONS`, newest first, each `{"TIME_PERIOD": "2026/08", "OBS_VALUE":
  "145.46199999999999000000", ...}`: values as text with twenty decimals.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from domains.coffee.config import InpcConfig
from mlops_core.data.api import ApiClient
from mlops_core.data.extract import RawArtifact, store_payload

logger = logging.getLogger(__name__)


def ingest_index(
    client: ApiClient, config: InpcConfig, token: str, raw_dir: Path, now: datetime | None = None
) -> RawArtifact:
    """Store the index's monthly observations, oldest first, as one raw JSON document."""
    where = f"{config.indicator}/es/{config.area}/false/{config.source}/2.0"
    endpoint = f"{config.base_url}/INDICATOR/{where}"
    payload = client.get_json(
        f"{endpoint}/{token}?type=json",
        cache_key=f"inpc:{config.indicator}:{config.area}:{config.source}",
    )
    series = payload["Series"][0]
    if series["INDICADOR"] != config.indicator or not series["OBSERVATIONS"]:
        raise RuntimeError(f"INEGI answered for {series['INDICADOR']} with no observations")
    document: dict[str, Any] = {
        "indicator": series["INDICADOR"],
        "unit": series["UNIT"],
        "frequency": series["FREQ"],
        "last_update": series["LASTUPDATE"],
        # Oldest first, and only what is read: an identical answer hashes alike.
        "observations": sorted(
            ({"period": o["TIME_PERIOD"], "value": o["OBS_VALUE"]} for o in series["OBSERVATIONS"]),
            key=lambda observation: observation["period"],
        ),
    }
    observations = document["observations"]
    logger.info(
        "INPC %s: %d months, %s to %s",
        config.indicator,
        len(observations),
        observations[0]["period"],
        observations[-1]["period"],
    )
    body = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return store_payload(config.name, config.filename, body, raw_dir, endpoint, now)


def to_frame(document: dict[str, Any]) -> pl.DataFrame:
    """The stored observations as rows of text: `period` ("2026/08") and `value`."""
    return pl.DataFrame(document["observations"], schema={"period": pl.String, "value": pl.String})
