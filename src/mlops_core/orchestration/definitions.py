"""Dagster assets, one graph per installed domain.

The orchestrator is a thin layer: every asset calls the same function the CLI calls,
so nothing here is required to run the pipelines. Installing a package under `domains/`
adds a whole graph, which is how the framework proves it is domain-parameterized.

Assets manage their own storage (immutable Parquet partitions), so they return
`MaterializeResult` metadata instead of handing values to an IO manager. Each of the
domain's models gets its own three assets, named after it.
"""

from collections.abc import Sequence
from pathlib import Path

from dagster import (
    AssetCheckResult,
    AssetChecksDefinition,
    AssetsDefinition,
    AssetSelection,
    Definitions,
    MaterializeResult,
    asset,
    asset_check,
    define_asset_job,
)

from mlops_core.adapter import DomainAdapter, available_domains, load_adapter
from mlops_core.config import ModelConfig, Settings
from mlops_core.data.clean import build_clean
from mlops_core.data.documents import fetch_documents
from mlops_core.data.extract import extract_all, http_client
from mlops_core.data.validate import validate_raw
from mlops_core.ml.features import build_features
from mlops_core.ml.predict import batch_predict
from mlops_core.ml.train import train_model
from mlops_core.storage import read_table

# Assets write their own Parquet, so they hand Dagster metadata, not a value.
Materialized = MaterializeResult[None]


def pipeline_assets(adapter: DomainAdapter) -> dict[str, list[str]]:
    """Asset names per pipeline, mirroring `mlops data run` and `mlops ml run`. The model
    tables keep the domain's own names, so the graph reads in its vocabulary."""
    return {
        "data": ["raw_sources", "clean_tables"],
        "ml": [name for model in adapter.config.models for name in model_asset_names(model)],
    }


def model_asset_names(model: ModelConfig) -> list[str]:
    return [model.features_table, f"{model.name}_model", model.predictions_table]


def domain_assets(adapter: DomainAdapter, settings: Settings) -> list[AssetsDefinition]:
    config = adapter.config
    data_dir = settings.data_dir / config.name
    prefix = [config.name]
    group = config.name

    @asset(name="raw_sources", key_prefix=prefix, group_name=group)
    def raw_sources() -> Materialized:
        """Every source downloaded untransformed, with an ingestion manifest.

        Files and APIs alike: the orchestrator runs the same extraction the CLI runs,
        so a source cannot be one that only arrives when a human types the command.
        """
        with http_client() as client:
            artifacts = extract_all(config, data_dir / "raw", client)
            api = adapter.extract(data_dir, client)
            corpus, absent = fetch_documents(config.documents, data_dir, client)
        artifacts |= api.artifacts | corpus
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
    for model in config.models:
        built += model_assets(adapter, model, settings, clean_tables)
    return built


def model_assets(
    adapter: DomainAdapter, model: ModelConfig, settings: Settings, clean_tables: AssetsDefinition
) -> list[AssetsDefinition]:
    """One model's features, training and batch scores, downstream of the clean layer."""
    config = adapter.config
    data_dir = settings.data_dir / config.name
    prefix, group = [config.name], config.name
    features_name, model_name, predictions_name = model_asset_names(model)

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
        """Batch scores for every row of the feature table."""
        path = batch_predict(config, model.name, data_dir, settings.mlflow_tracking_uri)
        return MaterializeResult(metadata={"path": str(path)})

    return [features, trained_model, predictions]


def domain_checks(
    adapter: DomainAdapter, settings: Settings, assets: list[AssetsDefinition]
) -> list[AssetChecksDefinition]:
    config = adapter.config
    data_dir = settings.data_dir / config.name
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
        adapters = [load_adapter(name) for name in available_domains()]
    assets: list[AssetsDefinition] = []
    checks: list[AssetChecksDefinition] = []
    for adapter in adapters:
        domain = domain_assets(adapter, settings)
        assets += domain
        checks += domain_checks(adapter, settings, domain)
    # One job per pipeline, mirroring `mlops data run` and `ml run`.
    jobs = [
        define_asset_job(
            name=f"{adapter.config.name}_{pipeline}",
            selection=AssetSelection.assets(*[[adapter.config.name, name] for name in names]),
        )
        for adapter in adapters
        for pipeline, names in pipeline_assets(adapter).items()
    ]
    return Definitions(assets=assets, asset_checks=checks, jobs=jobs)


defs = build_definitions()
