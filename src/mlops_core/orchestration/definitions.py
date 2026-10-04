"""Dagster assets, one graph per installed domain.

The orchestrator is a thin layer: every asset calls the same function the CLI calls,
so nothing here is required to run the pipelines. Installing a package under `domains/`
adds a whole graph, which is how the framework proves it is domain-parameterized.

What can run on its own: the data pipeline on the domain's `schedule`; each model's
features, scores and drift when the data it reads changes (sensor `<domain>_new_data`);
and its training when the monitor calls for it on data it was not trained on (sensor
`<domain>_retrain`). The questions the sensors ask are `triggers`' pure functions. All of
it starts off: the project runs by hand, and a deployment that should run on its own
turns it on with `MLOPS_AUTOMATE=true` (or each one in the UI).

A source whose history is its downloads (`accumulate`) also gets a daily-partitioned
asset, `<source>_reads`: a partition is a day it was read, in the schedule's timezone -
whether the read brought something new or found what the day before had - and
materializing one checks that day's reads against the contract, each on its own. The
sensor `<domain>_reads` asks for a day once for each new download, so the partitions
show which days were read and which were not - the history such a source can never be
asked for again - and a backfill re-checks the days read, and says of each day nobody
read that it cannot be read now.

Assets manage their own storage (immutable Parquet partitions), so they return
`MaterializeResult` metadata instead of handing values to an IO manager. Each of the
domain's models gets its own three assets, named after it.
"""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import mlflow
from dagster import (
    AssetCheckResult,
    AssetChecksDefinition,
    AssetExecutionContext,
    AssetKey,
    AssetsDefinition,
    AssetSelection,
    DailyPartitionsDefinition,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    Failure,
    MaterializeResult,
    RunRequest,
    ScheduleDefinition,
    SensorDefinition,
    SensorEvaluationContext,
    SkipReason,
    asset,
    asset_check,
    define_asset_job,
    sensor,
)

from mlops_core.adapter import DomainAdapter, available_tenants, load_adapter
from mlops_core.config import DomainConfig, ModelConfig, Settings
from mlops_core.data.clean import build_clean
from mlops_core.data.documents import fetch_documents
from mlops_core.data.extract import checks_by_day, extract_all, http_client, ingestions
from mlops_core.data.validate import validate_raw, validate_read
from mlops_core.ml.features import build_features
from mlops_core.ml.predict import batch_predict
from mlops_core.ml.registry import NoChampion
from mlops_core.ml.train import train_model
from mlops_core.monitoring.drift import monitor_model
from mlops_core.orchestration.triggers import new_data, retraining_due
from mlops_core.storage import read_table

if TYPE_CHECKING:  # not exported by dagster; only its name is needed
    from dagster._core.definitions.unresolved_asset_job_definition import (
        UnresolvedAssetJobDefinition,
    )

# Assets write their own Parquet, so they hand Dagster metadata, not a value.
Materialized = MaterializeResult[None]
# How often the sensors look: the data changes once a day at most.
SENSOR_SECONDS = 600
# The calendar of a domain without a schedule: a read's day has to be some zone's.
DEFAULT_TIMEZONE = "UTC"


def _tenant(config: DomainConfig) -> str:
    """A domain's or subdomain's name where Dagster wants an identifier: its assets' key
    prefix, its group, its jobs and sensors (`<domain>`, `<domain>__<subdomain>`)."""
    return config.tenant.replace("/", "__")


def schedule_status(settings: Settings) -> DefaultScheduleStatus:
    """On only where the deployment says so: by hand is the default."""
    return DefaultScheduleStatus.RUNNING if settings.automate else DefaultScheduleStatus.STOPPED


def sensor_status(settings: Settings) -> DefaultSensorStatus:
    return DefaultSensorStatus.RUNNING if settings.automate else DefaultSensorStatus.STOPPED


def pipeline_assets(adapter: DomainAdapter) -> dict[str, list[str]]:
    """Asset names per pipeline, mirroring `mlops data run` and `mlops ml run`. The model
    tables keep the domain's own names, so the graph reads in its vocabulary."""
    return {
        "data": ["raw_sources", "clean_tables"],
        "ml": [name for model in adapter.config.models for name in model_asset_names(model)],
    }


def model_asset_names(model: ModelConfig) -> list[str]:
    """A model's assets, in order: features, model, batch scores, and their drift."""
    return [
        model.features_table,
        f"{model.name}_model",
        model.predictions_table,
        f"{model.name}_drift",
    ]


def domain_assets(adapter: DomainAdapter, settings: Settings) -> list[AssetsDefinition]:
    config = adapter.config
    data_dir = settings.data_dir / config.home
    prefix = [_tenant(config)]
    group = _tenant(config)

    @asset(name="raw_sources", key_prefix=prefix, group_name=group)
    def raw_sources() -> Materialized:
        """Every source downloaded untransformed, with an ingestion manifest.

        Files and APIs alike: the orchestrator runs the same extraction the CLI runs,
        so a source cannot be one that only arrives when a human types the command.
        """
        with http_client() as client:
            files = extract_all(config, data_dir / "raw", client)
            api = adapter.extract(data_dir, client)
            corpus, absent = fetch_documents(
                config.documents,
                data_dir,
                client,
                refresh_hours=config.corpus.refresh_hours if config.corpus else None,
            )
        if files.failed:  # after every other source is stored: they are not lost
            raise RuntimeError(
                "Could not download: "
                + "; ".join(f"{name} ({why})" for name, why in files.failed.items())
            )
        artifacts = files.artifacts | api.artifacts | corpus
        skipped = api.skipped | absent
        return MaterializeResult(
            metadata={
                "sources": len(artifacts),
                "bytes": sum(a.manifest.size_bytes for a in artifacts.values()),
                # A skipped source is a shorter run, not a failed one; say which and why.
                "skipped": ", ".join(f"{n} ({w})" for n, w in skipped.items()),
            }
        )

    @asset(name="clean_tables", key_prefix=prefix, group_name=group, deps=[raw_sources])
    def clean_tables() -> Materialized:
        """Canonical, model-agnostic tables. Contract violations stop the run."""
        paths = build_clean(adapter, data_dir)
        return MaterializeResult(metadata={name: str(path) for name, path in paths.items()})

    built = [raw_sources, clean_tables]
    built += [read_assets(adapter, name, settings, raw_sources) for name in config.accumulate]
    for model in config.models:
        built += model_assets(adapter, model, settings, clean_tables)
    return built


def reads_timezone(adapter: DomainAdapter) -> str:
    schedule = adapter.config.schedule
    return schedule.timezone if schedule else DEFAULT_TIMEZONE


def read_days(adapter: DomainAdapter, name: str, settings: Settings) -> DailyPartitionsDefinition:
    """A partition a day, from the first day the source was read (today, if it never
    was) through today: a read made this morning is a partition before the day ends."""
    raw_dir = settings.data_dir / adapter.config.home / "raw"
    timezone = reads_timezone(adapter)
    days = sorted(checks_by_day(raw_dir, name, timezone))
    first = days[0] if days else datetime.now(ZoneInfo(timezone)).date().isoformat()
    return DailyPartitionsDefinition(start_date=first, timezone=timezone, end_offset=1)


def read_assets(
    adapter: DomainAdapter, name: str, settings: Settings, raw_sources: AssetsDefinition
) -> AssetsDefinition:
    """`<source>_reads`: one partition per day a source whose history is its downloads
    was read."""
    config = adapter.config
    raw_dir = settings.data_dir / config.home / "raw"
    timezone = reads_timezone(adapter)

    @asset(
        name=f"{name}_reads",
        key_prefix=[_tenant(config)],
        group_name=_tenant(config),
        partitions_def=read_days(adapter, name, settings),
        deps=[raw_sources],
    )
    def reads(context: AssetExecutionContext) -> Materialized:
        """What the source showed on the day: each read the day's downloads left or
        found, checked against its contract on its own."""
        day = context.partition_key
        downloads = checks_by_day(raw_dir, name, timezone).get(day, [])
        if not downloads:
            raise Failure(
                f"{name} was not read on {day}. It shows only what it holds when it is "
                "read: a day nobody read cannot be downloaded now, and the next read "
                "brings only what the source shows then."
            )
        stored = {artifact.partition.name: artifact for artifact in ingestions(raw_dir, name)}
        shown = list(dict.fromkeys(check.partition for check in downloads))  # in order, once
        checked = [validate_read(adapter, name, stored[partition]) for partition in shown]
        return MaterializeResult(
            metadata={
                "downloads": len(downloads),
                "new": sum(check.changed for check in downloads),
                "rows": sum(read.frame.height for read in checked),
                "latest": shown[-1],
            }
        )

    return reads


def model_assets(
    adapter: DomainAdapter, model: ModelConfig, settings: Settings, clean_tables: AssetsDefinition
) -> list[AssetsDefinition]:
    """One model's features, training and batch scores, downstream of the clean layer."""
    config = adapter.config
    data_dir = settings.data_dir / config.home
    prefix, group = [_tenant(config)], _tenant(config)
    features_name, model_name, predictions_name, drift_name = model_asset_names(model)

    @asset(name=features_name, key_prefix=prefix, group_name=group, deps=[clean_tables])
    def features() -> Materialized:
        """Model-ready table: the items enriched with the context they may see."""
        path = build_features(adapter, model.name, data_dir)
        return MaterializeResult(metadata={"path": str(path)})

    @asset(name=model_name, key_prefix=prefix, group_name=group, deps=[features])
    def trained_model() -> Materialized:
        """A tuned, tracked model; promoted to champion only if it passes the gate."""
        result = train_model(config, model.name, data_dir, settings.mlflow_tracking_uri)
        return MaterializeResult(
            metadata={"version": result.model_version, "promoted": str(result.promoted)}
            | {k: round(v, 4) for k, v in result.metrics.items()}
        )

    @asset(name=predictions_name, key_prefix=prefix, group_name=group, deps=[trained_model])
    def predictions() -> Materialized:
        """Batch scores for every row of the feature table - once a version has passed
        the gate; until then the run says so instead of failing."""
        try:
            path = batch_predict(config, model.name, data_dir, settings.mlflow_tracking_uri)
        except NoChampion:
            return MaterializeResult(metadata={"skipped": "no version has passed the gate"})
        return MaterializeResult(metadata={"path": str(path)})

    @asset(name=drift_name, key_prefix=prefix, group_name=group, deps=[predictions])
    def drift() -> Materialized:
        """The newest period against the earlier ones, and whether to retrain."""
        result = monitor_model(config, model.name, data_dir, settings.mlflow_tracking_uri)
        if result is None:
            return MaterializeResult(metadata={"skipped": "one period only"})
        return MaterializeResult(
            metadata={
                "current": result.current,
                "drifted_share": round(result.drifted_share, 4),
                "retrain": str(result.retrain),
                "reasons": "; ".join(result.reasons),
            }
        )

    return [features, trained_model, predictions, drift]


def domain_checks(
    adapter: DomainAdapter, settings: Settings, assets: list[AssetsDefinition]
) -> list[AssetChecksDefinition]:
    config = adapter.config
    data_dir = settings.data_dir / config.home
    by_name = {a.key.path[-1]: a for a in assets}

    @asset_check(asset=by_name["raw_sources"], name="sources_match_their_contracts")
    def sources_match_their_contracts() -> AssetCheckResult:
        validated = validate_raw(adapter, data_dir / "raw")
        return AssetCheckResult(
            passed=True, metadata={name: s.frame.height for name, s in validated.items()}
        )

    checks = [sources_match_their_contracts]
    for model in config.models:
        checks.append(leakage_check(model, by_name[model.features_table], data_dir))
    return checks


def leakage_check(
    model: ModelConfig, features: AssetsDefinition, data_dir: Path
) -> AssetChecksDefinition:
    """The model's feature table holds none of its leaking columns."""

    @asset_check(asset=features, name="no_leaking_columns")
    def no_leaking_columns() -> AssetCheckResult:
        columns = set(read_table(data_dir / "features" / model.features_table).columns)
        leaked = sorted(columns & set(model.spec.leakage))
        return AssetCheckResult(passed=not leaked, metadata={"leaked": ", ".join(leaked)})

    return no_leaking_columns


def build_definitions(
    adapters: Sequence[DomainAdapter] | None = None, settings: Settings | None = None
) -> Definitions:
    """Every installed domain's graph, or the ones given."""
    settings = settings or Settings()
    if adapters is None:
        adapters = [load_adapter(name) for name in available_tenants()]
    assets: list[AssetsDefinition] = []
    checks: list[AssetChecksDefinition] = []
    for adapter in adapters:
        domain = domain_assets(adapter, settings)
        assets += domain
        checks += domain_checks(adapter, settings, domain)
    # One job per pipeline, mirroring `mlops data run` and `ml run`, and one per source
    # whose history is its downloads.
    jobs: list[UnresolvedAssetJobDefinition] = [
        define_asset_job(
            name=f"{_tenant(adapter.config)}_{pipeline}",
            selection=AssetSelection.assets(*[[_tenant(adapter.config), name] for name in names]),
        )
        for adapter in adapters
        for pipeline, names in pipeline_assets(adapter).items()
    ]
    read_jobs = {_tenant(adapter.config): reads_jobs(adapter) for adapter in adapters}
    jobs += [job for domain_jobs in read_jobs.values() for job in domain_jobs]
    schedules = [
        ScheduleDefinition(
            name=f"{_tenant(adapter.config)}_daily_data",
            job_name=f"{_tenant(adapter.config)}_data",
            cron_schedule=adapter.config.schedule.data,
            execution_timezone=adapter.config.schedule.timezone,
            default_status=schedule_status(settings),
        )
        for adapter in adapters
        if adapter.config.schedule is not None
    ]
    sensors = [
        sensor
        for adapter in adapters
        for sensor in [
            *model_sensors(adapter, settings),
            *read_sensors(adapter, settings, read_jobs[_tenant(adapter.config)]),
        ]
    ]
    return Definitions(
        assets=assets, asset_checks=checks, jobs=jobs, schedules=schedules, sensors=sensors
    )


def model_sensors(adapter: DomainAdapter, settings: Settings) -> list[SensorDefinition]:
    """Score and monitor a model when its data changes; retrain it when the monitor calls
    for it on data it has not learned from. Each request is keyed by the data version, so
    one change starts one run."""
    config = adapter.config
    data_dir = settings.data_dir / config.home
    job = f"{_tenant(config)}_ml"

    def keys(model: str, *assets: str) -> list[AssetKey]:
        names = dict(zip(("features", "model", "predictions", "drift"),
                         model_asset_names(config.model_named(model)), strict=True))  # fmt: skip
        return [AssetKey([_tenant(config), names[asset]]) for asset in assets]

    @sensor(
        name=f"{_tenant(config)}_new_data",
        job_name=job,
        minimum_interval_seconds=SENSOR_SECONDS,
        default_status=sensor_status(settings),
    )
    def on_new_data(context: SensorEvaluationContext):  # type: ignore[no-untyped-def]
        """New data for a model: rebuild its features, score them with the champion, and
        compare the newest period with the ones before."""
        due = new_data(adapter, data_dir)
        if not due:
            yield SkipReason("every model's features come from the newest data")
        for model, version in due.items():
            yield RunRequest(
                run_key=f"{model}:{version}:score",
                asset_selection=keys(model, "features", "predictions", "drift"),
                tags={"model": model, "data_version": version},
            )

    @sensor(
        name=f"{_tenant(config)}_retrain",
        job_name=job,
        minimum_interval_seconds=SENSOR_SECONDS,
        default_status=sensor_status(settings),
    )
    def on_drift(context: SensorEvaluationContext):  # type: ignore[no-untyped-def]
        """The monitor calls for retraining on data no training run has seen: train, let
        the gate decide, and score and compare again."""
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        due = retraining_due(config, data_dir)
        if not due:
            yield SkipReason("no retraining due on data not trained on")
        for model, version in due.items():
            yield RunRequest(
                run_key=f"{model}:{version}:retrain",
                asset_selection=keys(model, "model", "predictions", "drift"),
                tags={"model": model, "data_version": version},
            )

    return [on_new_data, on_drift]


def reads_job(adapter: DomainAdapter, source: str) -> str:
    return f"{_tenant(adapter.config)}_{source}_reads"


def reads_jobs(adapter: DomainAdapter) -> "list[UnresolvedAssetJobDefinition]":
    """A job per source whose history is its downloads: a partitioned asset runs a day at
    a time, on its own partitions."""
    return [
        define_asset_job(
            name=reads_job(adapter, source),
            selection=AssetSelection.assets([_tenant(adapter.config), f"{source}_reads"]),
        )
        for source in adapter.config.accumulate
    ]


def read_sensors(
    adapter: DomainAdapter, settings: Settings, jobs: "list[UnresolvedAssetJobDefinition]"
) -> list[SensorDefinition]:
    """Check each day a source whose history is its downloads was read, once per read.
    None when the domain has no such source."""
    config = adapter.config
    if not config.accumulate:
        return []
    raw_dir = settings.data_dir / config.home / "raw"
    timezone = reads_timezone(adapter)

    @sensor(
        name=f"{_tenant(config)}_reads",
        jobs=jobs,
        minimum_interval_seconds=SENSOR_SECONDS,
        default_status=sensor_status(settings),
    )
    def on_reads(context: SensorEvaluationContext):  # type: ignore[no-untyped-def]
        """A day with a download not checked yet. The key names the day's latest
        download, so a second one the same day is checked too, and nothing twice."""
        requested = False
        for source in config.accumulate:
            for day, downloads in checks_by_day(raw_dir, source, timezone).items():
                requested = True
                yield RunRequest(
                    run_key=f"{source}:{day}:{downloads[-1].checked_at.isoformat()}",
                    job_name=reads_job(adapter, source),
                    partition_key=day,
                )
        if not requested:
            yield SkipReason("no source whose history is its downloads has been read yet")

    return [on_reads]


defs = build_definitions()
