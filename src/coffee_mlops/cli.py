"""Command-line entry point. Every command is parameterized by domain."""

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from coffee_mlops.catalog import connect
from coffee_mlops.clean import build_clean
from coffee_mlops.config import DomainConfig, Settings, load_domain_config
from coffee_mlops.extract import extract_all, http_client
from coffee_mlops.features import build_features
from coffee_mlops.validate import validate_raw

app = typer.Typer(no_args_is_help=True, add_completion=False)

Domain = Annotated[
    str, typer.Option("--domain", "-d", help="Domain config in configs/<domain>.yaml")
]


def _data_dir(config: DomainConfig) -> Path:
    return Settings().data_dir / config.name


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


@app.command()
def extract(domain: Domain = "coffee") -> None:
    """Download every source of the domain to the raw layer."""
    config = load_domain_config(domain)
    with http_client() as client:
        artifacts = extract_all(config, _data_dir(config) / "raw", client)
    for name, artifact in artifacts.items():
        typer.echo(f"{name}: {artifact.path} ({artifact.manifest.size_bytes:,} bytes)")


@app.command()
def validate(domain: Domain = "coffee") -> None:
    """Check the latest raw ingestion of every source against its contract."""
    config = load_domain_config(domain)
    for name, source in validate_raw(config, _data_dir(config) / "raw").items():
        typer.echo(f"{name}: {source.frame.height:,} rows valid ({source.artifact.partition.name})")


@app.command()
def clean(domain: Domain = "coffee") -> None:
    """Build the clean layer from the latest validated raw data."""
    config = load_domain_config(domain)
    for table, path in build_clean(config, _data_dir(config)).items():
        typer.echo(f"{table}: {path}")


@app.command()
def features(domain: Domain = "coffee") -> None:
    """Build the model-ready feature table from the latest clean layer."""
    config = load_domain_config(domain)
    typer.echo(f"review_features: {build_features(config, _data_dir(config))}")


@app.command()
def sql(
    query: Annotated[str, typer.Argument(help="e.g. 'SELECT * FROM clean.coffee_reviews'")],
    domain: Domain = "coffee",
) -> None:
    """Run SQL over the latest partition of every layer."""
    config = load_domain_config(domain)
    typer.echo(connect(_data_dir(config)).sql(query))


@app.command()
def train(domain: Domain = "coffee") -> None:
    """Tune and train a model, track it in MLflow, promote it if it passes the quality gate."""
    # Imported here: MLflow and LightGBM take seconds to import and no other command needs them.
    from coffee_mlops.train import train_model

    config = load_domain_config(domain)
    result = train_model(config, _data_dir(config), Settings().mlflow_tracking_uri)
    metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(result.metrics.items()))
    status = "promoted to champion" if result.promoted else "not promoted"
    typer.echo(f"{config.training.registered_model} v{result.model_version}: {status}")
    typer.echo(f"run {result.run_id}: {metrics}")
