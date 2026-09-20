"""Champion model loading.

MLflow is the source of truth, but the API keeps a local copy of what it loaded: a
restart while the tracking server is down should keep serving the last known model
instead of failing. `source` always says which one is in memory.
"""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.environment_variables import (
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

CHAMPION = "champion"
CACHED_MODEL = "model"
CACHED_METADATA = "cached.json"


@dataclass(frozen=True)
class ServedModel:
    model: Any
    version: str
    source: str  # "registry" or "cache"


def load_champion(
    registered_model: str,
    tracking_uri: str,
    cache_dir: Path,
    probe_retries: int = 1,
    probe_timeout_s: int = 10,
) -> ServedModel:
    """Load the champion from the registry and refresh the cache; fall back to the cache."""
    # MLflow defaults to 7 retries with exponential backoff and a 120 s timeout, so a
    # dead registry would take minutes to fall back to the cache instead of seconds.
    os.environ[MLFLOW_HTTP_REQUEST_MAX_RETRIES.name] = str(probe_retries)
    os.environ[MLFLOW_HTTP_REQUEST_TIMEOUT.name] = str(probe_timeout_s)
    try:
        return _from_registry(registered_model, tracking_uri, cache_dir)
    except Exception as unreachable:  # network, auth, or no champion registered yet
        logger.warning("Registry unavailable (%s); falling back to the cache", unreachable)
        return _from_cache(cache_dir)


def _from_registry(registered_model: str, tracking_uri: str, cache_dir: Path) -> ServedModel:
    mlflow.set_tracking_uri(tracking_uri)
    version = mlflow.MlflowClient().get_model_version_by_alias(registered_model, CHAMPION)
    # Download once and load from the cache, so what is served is exactly what is cached.
    _refresh_cache(f"models:/{registered_model}@{CHAMPION}", version.version, cache_dir)
    model = mlflow.sklearn.load_model(str(cache_dir / CACHED_MODEL))
    logger.info("Serving %s v%s from the registry", registered_model, version.version)
    return ServedModel(model, str(version.version), "registry")  # MLflow returns an int


def _from_cache(cache_dir: Path) -> ServedModel:
    metadata_path = cache_dir / CACHED_METADATA
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"No champion in the registry and no cached model in {cache_dir}: train one first"
        )
    version = str(json.loads(metadata_path.read_text())["version"])
    logger.warning("Serving cached v%s: it may be behind the registry", version)
    return ServedModel(mlflow.sklearn.load_model(str(cache_dir / CACHED_MODEL)), version, "cache")


def _refresh_cache(model_uri: str, version: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as staging:
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=staging)
        target = cache_dir / CACHED_MODEL
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(downloaded, target)
    # Written last: a half-copied model has no metadata and is never served.
    (cache_dir / CACHED_METADATA).write_text(json.dumps({"version": version}))
