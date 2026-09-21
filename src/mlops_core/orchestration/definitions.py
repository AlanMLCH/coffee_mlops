"""Dagster assets, one graph per domain config.

The orchestrator is a thin layer: every asset calls the same function the CLI calls,
so nothing here is required to run the pipelines. Adding `configs/<domain>.yaml` adds
a whole graph, which is how the framework proves it is domain-parameterized.

Assets manage their own storage (immutable Parquet partitions), so they return
`MaterializeResult` metadata instead of handing values to an IO manager.
"""

from collections.abc import Iterator
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

from mlops_core.config import CONFIGS_DIR, DomainConfig, Settings, load_domain_config
from mlops_core.data.clean import build_clean
from mlops_core.data.extract import extract_all, http_client
from mlops_core.data.sources import extract_api_sources
from mlops_core.data.validate import validate_raw
from mlops_core.ml.features import build_features
from mlops_core.ml.predict import batch_predict
from mlops_core.ml.train import train_model
from mlops_core.storage import read_table

# Assets write their own Parquet, so they hand Dagster metadata, not a value.
Materialized = MaterializeResult[None]

DATA_ASSETS = ["raw_sources", "clean_tables"]
ML_ASSETS = ["review_features", "trained_model", "review_predictions"]


def domain_assets(config: DomainConfig, settings: Settings) -> list[AssetsDefinition]:
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
            api = extract_api_sources(config, settings, data_dir, client)
        artifacts |= api.artifacts
        return MaterializeResult(
            metadata={
                "sources": len(artifacts),
                "bytes": sum(a.manifest.size_bytes for a in artifacts.values()),
                # A skipped source is a shorter run, not a failed one; say which and why.
                "skipped": ", ".join(f"{n} ({w})" for n, w in api.skipped.items()),
            }
        )

    @asset(name="clean_tables", key_prefix=prefix, group_name=group, deps=[raw_sources])
    def clean_tables() -> Materialized:
        """Canonical, model-agnostic tables. Contract violations stop the run."""
        paths = build_clean(config, data_dir)
        return MaterializeResult(metadata={name: str(path) for name, path in paths.items()})

    @asset(name="review_features", key_prefix=prefix, group_name=group, deps=[clean_tables])
    def review_features() -> Materialized:
        """Model-ready table: clean items plus point-in-time market context."""
        return MaterializeResult(metadata={"path": str(build_features(config, data_dir))})

    @asset(name="trained_model", key_prefix=prefix, group_name=group, deps=[review_features])
    def trained_model() -> Materialized:
        """A tuned, tracked model; promoted to champion only if it passes the gate."""
        result = train_model(config, data_dir, settings.mlflow_tracking_uri)
        return MaterializeResult(
            metadata={"version": result.model_version, "promoted": str(result.promoted)}
            | {k: round(v, 4) for k, v in result.metrics.items()}
        )

    @asset(name="review_predictions", key_prefix=prefix, group_name=group, deps=[trained_model])
    def review_predictions() -> Materialized:
        """Batch scores for every row of the feature table."""
        path = batch_predict(config, data_dir, settings.mlflow_tracking_uri)
        return MaterializeResult(metadata={"path": str(path)})

    return [raw_sources, clean_tables, review_features, trained_model, review_predictions]


def domain_checks(
    config: DomainConfig, settings: Settings, assets: list[AssetsDefinition]
) -> list[AssetChecksDefinition]:
    data_dir = settings.data_dir / config.name
    by_name = {a.key.path[-1]: a for a in assets}

    @asset_check(asset=by_name["raw_sources"], name="sources_match_their_contracts")
    def sources_match_their_contracts() -> AssetCheckResult:
        validated = validate_raw(config, data_dir / "raw")
        return AssetCheckResult(
            passed=True, metadata={name: s.frame.height for name, s in validated.items()}
        )

    @asset_check(asset=by_name["review_features"], name="no_leaking_columns")
    def no_leaking_columns() -> AssetCheckResult:
        columns = set(read_table(data_dir / "features" / "review_features").columns)
        leaked = sorted(columns & set(config.model.leakage))
        return AssetCheckResult(passed=not leaked, metadata={"leaked": ", ".join(leaked)})

    return [sources_match_their_contracts, no_leaking_columns]


def build_definitions(
    configs_dir: Path = CONFIGS_DIR, settings: Settings | None = None
) -> Definitions:
    settings = settings or Settings()
    domains = list(_domains(configs_dir))
    assets: list[AssetsDefinition] = []
    checks: list[AssetChecksDefinition] = []
    for config in domains:
        domain = domain_assets(config, settings)
        assets += domain
        checks += domain_checks(config, settings, domain)
    # One job per pipeline, mirroring `mlops data run` and `ml run`.
    jobs = [
        define_asset_job(
            name=f"{config.name}_{pipeline}",
            selection=AssetSelection.assets(*[[config.name, name] for name in names]),
        )
        for config in domains
        for pipeline, names in (("data", DATA_ASSETS), ("ml", ML_ASSETS))
    ]
    return Definitions(assets=assets, asset_checks=checks, jobs=jobs)


def _domains(configs_dir: Path) -> Iterator[DomainConfig]:
    for path in sorted(configs_dir.glob("*.yaml")):
        yield load_domain_config(path.stem, configs_dir)


defs = build_definitions()
