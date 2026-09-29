"""What is ready and what is left to do: the data, the models, the services and the keys,
each checked where it lives, with the command that makes it ready.

`mlops status` exists because the explorer and the agent stand on many things at once -
built layers, a champion in the registry, the prediction API, a current index in the
vector store, two models in the local model server - and a missing one surfaces as an
error somewhere else, far from its cause. Every check here is independent: a service
that is down is a finding that says what to run, never a crash that hides the rest.

Like the CLI and the orchestrator, this module sees every pipeline; each import is made
where it is used, so the command stays light to load and says which extra is missing.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from mlops_core.adapter import DomainAdapter
from mlops_core.config import DomainConfig, Settings
from mlops_core.storage import TIMESTAMP_FORMAT, latest_partition

DATA, MODELS, SERVICES, KEYS = "data", "models", "services", "keys"
EXTRACT, CLEAN, ML, ANALYSIS = "make extract", "make clean-layer", "make ml", "make analysis"


@dataclass(frozen=True)
class Finding:
    """One thing checked. `ready` is True when nothing needs doing, False when something
    does (and `fix` says what), None for a note: a result to know, with nothing to run."""

    section: str
    name: str
    ready: bool | None
    detail: str
    fix: str = ""


def data_findings(
    adapter: DomainAdapter, data_dir: Path, now: datetime | None = None
) -> list[Finding]:
    """Each source's last download, each document, and whether every derived layer was
    built after the raw data it reads."""
    from mlops_core.data.extract import ingestions, last_checked

    config, now = adapter.config, now or datetime.now(UTC)
    raw = data_dir / "raw"
    found: list[Finding] = []
    refresh = {name: source.refresh_hours for name, source in config.sources.items()}
    # Credentials are optional: an API source is skipped while one it needs is missing, and
    # which one is the domain's to know. A never-downloaded API source is then a note.
    keyless = any(secret is None for secret in adapter.credentials().values())
    for name in [*config.sources, *adapter.json_readers()]:
        checked = last_checked(raw, name)
        if checked is None and keyless and name not in config.sources:
            detail = "never downloaded: an API source, skipped while its key is missing (keys)"
            found.append(Finding(DATA, name, None, detail))
            continue
        if checked is None:
            found.append(Finding(DATA, name, False, "never downloaded", EXTRACT))
            continue
        hours = refresh.get(name)
        due = hours is not None and now - checked >= timedelta(hours=hours)
        reads = len(ingestions(raw, name))
        kept = f", {reads} read{'s' * (reads != 1)} kept" if name in config.accumulate else ""
        detail = f"checked {ago(now - checked)} ago{kept}" + (" - due again" if due else "")
        found.append(Finding(DATA, name, not due, detail, EXTRACT if due else ""))
    inbox = data_dir / "inbox" / "documents"
    missing = [document for document in config.documents if not ingestions(raw, document.name)]
    if config.documents:
        detail = f"{len(config.documents) - len(missing)} of {len(config.documents)} read"
        by_hand = [d.inbox for d in missing if d.inbox]
        if by_hand:
            detail += f"; put {', '.join(by_hand)} in {inbox.as_posix()}"
        # A document only a person can fetch is optional: the corpus works without it.
        fetchable = [d for d in missing if not d.inbox]
        ready = None if missing and not fetchable else not missing
        found.append(Finding(DATA, "documents", ready, detail, EXTRACT if fetchable else ""))
    newest_raw = max((stamp(p) for p in raw.glob("*/*=*") if p.is_dir()), default=None)
    clean_tables = [*adapter.clean_contracts(), *config.corpus_tables]
    found.append(_layer(data_dir / "clean", clean_tables, newest_raw, CLEAN))
    found += _models_layers(adapter, data_dir)
    return found


def _layer(layer: Path, tables: list[str], newest_input: datetime | None, fix: str) -> Finding:
    """A layer is ready when every table is built, after the newest data it reads."""
    built = {table: latest_partition(layer / table) for table in tables}
    absent = sorted(table for table, partition in built.items() if partition is None)
    if absent:
        return Finding(DATA, layer.name, False, f"not built: {', '.join(absent)}", fix)
    oldest = min(stamp(partition) for partition in built.values() if partition is not None)
    if newest_input is not None and newest_input > oldest:
        return Finding(DATA, layer.name, False, "built before the newest download", fix)
    return Finding(DATA, layer.name, True, f"{len(tables)} tables, built {oldest:%Y-%m-%d %H:%M}")


def _models_layers(adapter: DomainAdapter, data_dir: Path) -> list[Finding]:
    """Each model's features and predictions, against the clean data they are built from,
    and the analysis against the clean layer."""
    from mlops_core.orchestration.triggers import new_data

    config = adapter.config
    found = []
    stale = new_data(adapter, data_dir)
    for model in config.models:
        features = latest_partition(data_dir / "features" / model.features_table)
        predictions = latest_partition(data_dir / "predictions" / model.predictions_table)
        if features is None:
            found.append(Finding(DATA, model.features_table, False, "not built", ML))
        elif model.name in stale:
            found.append(Finding(DATA, model.features_table, False, "older than its data", ML))
        else:
            found.append(Finding(DATA, model.features_table, True, "built from the current data"))
        if predictions is None:  # a model the gate never promoted has none, and that is right
            detail = "none: only a champion scores them, with make ml"
            found.append(Finding(DATA, model.predictions_table, None, detail))
    newest_clean = newest(data_dir / "clean")
    analysed = newest(data_dir / "analysis")  # its study tables: the figures keep no manifest
    if analysed is None:
        found.append(Finding(DATA, "analysis", False, "not run", ANALYSIS))
    elif newest_clean is not None and newest_clean > analysed:
        found.append(Finding(DATA, "analysis", False, "run before the last clean build", ANALYSIS))
    else:
        found.append(Finding(DATA, "analysis", True, f"run {analysed:%Y-%m-%d %H:%M}"))
    return found


def newest(layer: Path) -> datetime | None:
    """When the newest complete partition of any table in a layer was written."""
    partitions = [latest_partition(table) for table in layer.glob("*") if table.is_dir()]
    return max((stamp(p) for p in partitions if p is not None), default=None)


def model_findings(
    config: DomainConfig, data_dir: Path, champion: Callable[[str], str | None]
) -> list[Finding]:
    """Each model's champion - `champion(registered_model)` gives its version, None when
    the gate never promoted one - and whether a retraining the monitor asked for is due.
    Tracking must point at the registry."""
    from mlops_core.orchestration.triggers import retraining_due

    found = []
    for model in config.models:
        version = champion(model.training.registered_model)
        if version is None:
            found.append(Finding(MODELS, model.name, None, "no version has passed the gate"))
        else:
            found.append(Finding(MODELS, model.name, True, f"champion v{version}"))
    for name, version in retraining_due(config, data_dir).items():
        detail = f"drift calls for retraining on data {version[:12]}, not trained on yet"
        found.append(Finding(MODELS, f"{name} retraining", False, detail, "make retrain"))
    return found


def service_findings(
    settings: Settings, config: DomainConfig, data_dir: Path, http: httpx.Client
) -> list[Finding]:
    """The registry, the prediction API, the vector index and the local model server:
    reachable, and holding what the agent needs."""
    uri = settings.mlflow_tracking_uri
    registry = (
        _reachable("MLflow", f"{uri}/health", http, "make services-up PROFILE=ml")
        if uri.startswith(("http://", "https://"))
        else Finding(SERVICES, "MLflow", True, f"a local registry: {uri}")
    )
    return [registry, _api(settings, http), _index(settings, config, data_dir),
            *_local_models(settings, http)]  # fmt: skip


def key_findings(adapter: DomainAdapter) -> list[Finding]:
    """Each credential the domain reads: set or not, never its value. A missing one is a
    note, not a task: credentials are optional, and what needs one is skipped out loud."""
    missing = "missing: optional, what needs it is skipped (see .env.example)"
    return [
        Finding(KEYS, label, True if secret else None, "set" if secret else missing)
        for label, secret in adapter.credentials().items()
    ]


def _reachable(name: str, url: str, http: httpx.Client, fix: str) -> Finding:
    try:
        http.get(url).raise_for_status()
    except httpx.HTTPError as failed:
        return Finding(SERVICES, name, False, f"not answering at {url}: {first_line(failed)}", fix)
    return Finding(SERVICES, name, True, f"answering at {url}")


def _api(settings: Settings, http: httpx.Client) -> Finding:
    fix = "make services-up PROFILE=api"
    try:
        health: dict[str, Any] = http.get(f"{settings.api_url}/health").json()
    except (httpx.HTTPError, ValueError) as failed:
        return Finding(
            SERVICES, "prediction API", False, f"not answering: {first_line(failed)}", fix
        )
    served = health.get("models", {})
    missing = sorted(name for name, version in served.items() if version is None)
    detail = ", ".join(f"{name} v{version}" for name, version in served.items() if version)
    if missing:
        detail += f"; not served: {', '.join(missing)}"
    return Finding(SERVICES, "prediction API", True, detail or health.get("status", ""))


def _index(settings: Settings, config: DomainConfig, data_dir: Path) -> Finding:
    """The vector index, built from the chunks on disk (compared by their content)."""
    if not config.documents:
        return Finding(SERVICES, "index", None, "the domain lists no documents")
    fix = "make index"
    try:
        from qdrant_client import QdrantClient

        from mlops_core.config import CHUNKS_TABLE
        from mlops_core.rag.vectors import chunks_digest, index_metadata
        from mlops_core.storage import read_table
    except ImportError:
        return Finding(SERVICES, "index", None, "the rag extra is not installed")
    try:
        chunks = read_table(data_dir / "clean" / CHUNKS_TABLE)
    except FileNotFoundError:
        return Finding(SERVICES, "index", False, "no chunks yet", CLEAN)
    try:
        built = index_metadata(QdrantClient(url=settings.qdrant_url, timeout=5), config.name)
    except Exception as failed:  # unreachable, or no alias yet: both are qdrant's own errors
        return Finding(
            SERVICES, "index", False, first_line(failed), f"make services-up PROFILE=ai, {fix}"
        )
    if built.get("chunks_digest") != chunks_digest(chunks):
        return Finding(SERVICES, "index", False, "built from other chunks", fix)
    return Finding(SERVICES, "index", True, f"current: {chunks.height:,} chunks")


def _local_models(settings: Settings, http: httpx.Client) -> list[Finding]:
    """The generator and the embedding model the agent runs, pulled in the model server."""
    try:
        from mlops_core.agent.graph import AGENT_GENERATOR
        from mlops_core.rag.vectors import EMBEDDING_MODEL
    except ImportError:
        return [Finding(SERVICES, "local models", None, "the agent extra is not installed")]
    try:
        tags = http.get(f"{settings.ollama_url}/api/tags").json()
    except (httpx.HTTPError, ValueError) as failed:
        return [Finding(SERVICES, "local models", False, f"not answering: {first_line(failed)}",
                        "start Ollama")]  # fmt: skip
    pulled = {model["name"] for model in tags.get("models", [])}
    return [
        Finding(
            SERVICES,
            name,
            name in pulled,
            "pulled" if name in pulled else "not pulled",
            "" if name in pulled else f"ollama pull {name}",
        )
        for name in (AGENT_GENERATOR, EMBEDDING_MODEL)
    ]


# The order to run fixes in: keys and services before the data that needs them, the data
# before the models and the index built from it.
ORDER = (".env", "services-up", "Ollama", "ollama pull", EXTRACT, CLEAN, ML, "make retrain",
         ANALYSIS, "make index")  # fmt: skip


def to_do(findings: list[Finding]) -> dict[str, list[str]]:
    """Each command to run, with what it fixes, in the order they should run."""
    commands: dict[str, list[str]] = {}
    for finding in findings:
        if finding.ready is False and finding.fix:
            commands.setdefault(finding.fix, []).append(finding.name)

    def rank(command: str) -> int:
        return next((i for i, word in enumerate(ORDER) if word in command), len(ORDER))

    return dict(sorted(commands.items(), key=lambda item: rank(item[0])))


def stamp(partition: Path) -> datetime:
    """When a partition was written, from its name (`ingested_at=20260929T044001996244Z`)."""
    return datetime.strptime(partition.name.split("=", 1)[1], TIMESTAMP_FORMAT).replace(tzinfo=UTC)


def ago(age: timedelta) -> str:
    """ "3 days", "5 hours", "12 minutes": the largest whole unit."""
    for unit, seconds in (("day", 86_400), ("hour", 3_600), ("minute", 60)):
        count = int(age.total_seconds() // seconds)
        if count:
            return f"{count} {unit}{'s' if count > 1 else ''}"
    return "moments"


def first_line(error: BaseException) -> str:
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__


def by_section(findings: list[Finding]) -> Mapping[str, list[Finding]]:
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.section, []).append(finding)
    return grouped
