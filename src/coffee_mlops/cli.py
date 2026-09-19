"""Command-line entry point. Every command is parameterized by domain."""

import logging
from typing import Annotated

import typer

from coffee_mlops.config import Settings, load_domain_config
from coffee_mlops.extract import extract_all, http_client
from coffee_mlops.validate import validate_raw

app = typer.Typer(no_args_is_help=True, add_completion=False)

Domain = Annotated[
    str, typer.Option("--domain", "-d", help="Domain config in configs/<domain>.yaml")
]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
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
    raw_dir = Settings().data_dir / config.name / "raw"
    with http_client() as client:
        artifacts = extract_all(config, raw_dir, client)
    for name, artifact in artifacts.items():
        typer.echo(f"{name}: {artifact.path} ({artifact.manifest.size_bytes:,} bytes)")


@app.command()
def validate(domain: Domain = "coffee") -> None:
    """Check the latest raw ingestion of every source against its contract."""
    config = load_domain_config(domain)
    frames = validate_raw(config, Settings().data_dir / config.name / "raw")
    for name, frame in frames.items():
        typer.echo(f"{name}: {frame.height:,} rows valid")
