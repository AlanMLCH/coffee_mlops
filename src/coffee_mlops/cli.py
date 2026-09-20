"""Command-line entry point.

Two independent pipelines, one command group each. `data` produces the canonical
clean tables; `ml` consumes them from disk. Every step runs on its own; `run`
chains the steps of one pipeline when that is what you want.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from coffee_mlops.catalog import connect
from coffee_mlops.config import DomainConfig, Settings, load_domain_config
from coffee_mlops.data.clean import build_clean
from coffee_mlops.data.extract import extract_all, http_client
from coffee_mlops.data.validate import validate_raw

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="ETL: external sources -> clean tables.")
ml_app = typer.Typer(no_args_is_help=True, help="Model pipeline: clean tables -> model.")
app.add_typer(data_app, name="data")
app.add_typer(ml_app, name="ml")

Domain = Annotated[
    str, typer.Option("--domain", "-d", help="Domain config in configs/<domain>.yaml")
]


def _data_dir(config: DomainConfig) -> Path:
    return Settings().data_dir / config.name


@contextmanager
def _needs_extra(extra: str) -> Iterator[None]:
    """Each pipeline installs on its own, so say which extra is missing instead of
    dropping a traceback on someone who installed only the other pipeline."""
    try:
        yield
    except ModuleNotFoundError as missing:
        typer.echo(
            f"This command needs '{missing.name}', part of the '{extra}' extra. "
            f"Install it with: uv sync --extra {extra}",
            err=True,
        )
        raise typer.Exit(code=1) from missing


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    # Windows consoles default to cp1252: table borders and accented names need UTF-8.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs full request URLs at INFO: signed URLs and API tokens in query strings.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@data_app.command()
def extract(domain: Domain = "coffee") -> None:
    """Download every source of the domain to the raw layer."""
    config = load_domain_config(domain)
    with http_client() as client:
        artifacts = extract_all(config, _data_dir(config) / "raw", client)
    for name, artifact in artifacts.items():
        typer.echo(f"{name}: {artifact.path} ({artifact.manifest.size_bytes:,} bytes)")


@data_app.command()
def validate(domain: Domain = "coffee") -> None:
    """Check the latest raw ingestion of every source against its contract."""
    config = load_domain_config(domain)
    for name, source in validate_raw(config, _data_dir(config) / "raw").items():
        typer.echo(f"{name}: {source.frame.height:,} rows valid ({source.artifact.partition.name})")


@data_app.command()
def clean(domain: Domain = "coffee") -> None:
    """Build the clean layer from the latest validated raw data."""
    config = load_domain_config(domain)
    for table, path in build_clean(config, _data_dir(config)).items():
        typer.echo(f"{table}: {path}")


@data_app.command("run")
def data_run(domain: Domain = "coffee") -> None:
    """Whole ETL: extract, then validate and clean."""
    extract(domain)
    clean(domain)


@ml_app.command()
def features(domain: Domain = "coffee") -> None:
    """Build the model-ready feature table from the latest clean layer."""
    with _needs_extra("ml"):
        from coffee_mlops.ml.features import build_features

    config = load_domain_config(domain)
    typer.echo(f"review_features: {build_features(config, _data_dir(config))}")


@ml_app.command()
def train(domain: Domain = "coffee") -> None:
    """Tune and train a model, track it in MLflow, promote it if it passes the quality gate."""
    # Imported here: MLflow and LightGBM take seconds to import and no other command needs them.
    with _needs_extra("ml"):
        from coffee_mlops.ml.train import train_model

    config = load_domain_config(domain)
    result = train_model(config, _data_dir(config), Settings().mlflow_tracking_uri)
    metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(result.metrics.items()))
    status = "promoted to champion" if result.promoted else "not promoted"
    typer.echo(f"{config.training.registered_model} v{result.model_version}: {status}")
    typer.echo(f"run {result.run_id}: {metrics}")


@ml_app.command("run")
def ml_run(domain: Domain = "coffee") -> None:
    """Whole model pipeline: features, then train."""
    features(domain)
    train(domain)


@app.command()
def sql(
    query: Annotated[str, typer.Argument(help="e.g. 'SELECT * FROM clean.coffee_reviews'")],
    domain: Domain = "coffee",
) -> None:
    """Run SQL over the latest partition of every layer."""
    config = load_domain_config(domain)
    typer.echo(connect(_data_dir(config)).sql(query))
