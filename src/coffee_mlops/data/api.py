"""A polite HTTP client for the APIs stage 2 reads.

Three things every public API needs from a client, and none of them belong in the code
that knows about coffee:

- **A rate limit.** DENUE and Overpass are free services run for everyone; hammering
  them is both rude and the fastest way to get blocked.
- **Retries that discriminate.** A timeout, a 429 or a 500 are worth trying again; a 404
  or a bad token never are, and retrying them only wastes someone's capacity.
- **A cache on disk.** Re-running a pipeline must not re-download 99 pages. The cache is
  keyed by a *sanitised* identity supplied by the caller, never by the raw URL: DENUE
  carries its token in the URL path, and a cache keyed on that would write the
  credential into a file name.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def silence_request_urls() -> None:
    """Stop httpx from logging full request URLs at INFO.

    DENUE carries its token in the URL path, so one INFO line is a leaked credential.
    This lives next to the client, not in the CLI, because a safeguard that depends on
    which entry point you came through is not a safeguard: a script, a notebook or an
    orchestrator would each have to remember it.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ApiClient:
    """Wraps an `httpx.Client` with a rate limit, retries and an on-disk cache."""

    client: httpx.Client
    cache_dir: Path
    min_interval_s: float = 1.0
    max_attempts: int = 4
    backoff_s: float = 1.0
    # Injected so tests do not spend real seconds proving that waiting happens.
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    _last_request_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        silence_request_urls()

    def get_json(self, url: str, cache_key: str, headers: Mapping[str, str] | None = None) -> Any:
        """Fetch and parse JSON, from the cache when this exact request was made before.

        `cache_key` identifies the request *without any credential*: the caller knows
        which parts of the URL are secret, this class cannot.
        """
        cached = self._cache_path(cache_key)
        if cached.is_file():
            logger.debug("cache hit: %s", cache_key)
            return json.loads(cached.read_text(encoding="utf-8"))

        payload = self._fetch(url, headers)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _fetch(self, url: str, headers: Mapping[str, str] | None) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._wait_turn()
            try:
                response = self.client.get(url, headers=dict(headers or {}))
            except httpx.TransportError as unreachable:  # DNS, connection, timeout
                last_error = unreachable
                self._backoff(attempt, str(unreachable))
                continue
            if response.status_code in RETRYABLE_STATUS:
                last_error = httpx.HTTPStatusError(
                    f"{response.status_code} from the service",
                    request=response.request,
                    response=response,
                )
                self._backoff(attempt, f"status {response.status_code}")
                continue
            response.raise_for_status()  # 4xx: our request is wrong, retrying cannot fix it
            return response.json()
        raise RuntimeError(f"Giving up after {self.max_attempts} attempts") from last_error

    def _wait_turn(self) -> None:
        if self._last_request_at is not None:
            waited = self.now() - self._last_request_at
            if waited < self.min_interval_s:
                self.sleep(self.min_interval_s - waited)
        self._last_request_at = self.now()

    def _backoff(self, attempt: int, reason: str) -> None:
        if attempt < self.max_attempts:
            delay = self.backoff_s * 2 ** (attempt - 1)
            logger.warning("Retrying in %.1fs after %s (attempt %d)", delay, reason, attempt)
            self.sleep(delay)

    def _cache_path(self, cache_key: str) -> Path:
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"
