"""Command-line entry point.

Two independent pipelines, one command group each. `data` produces the canonical
clean tables; `ml` consumes them from disk. Every step runs on its own; `run`
chains the steps of one pipeline when that is what you want. Every command takes
`--domain`: the CLI knows the pipeline, the domain's adapter knows the rest.
"""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from mlops_core.adapter import DomainAdapter, load_adapter
from mlops_core.config import DomainConfig, Settings
from mlops_core.data.api import silence_request_urls
from mlops_core.data.clean import build_clean
from mlops_core.data.extract import extract_all, http_client
from mlops_core.data.validate import validate_raw
from mlops_core.provenance import REPO_ROOT
from mlops_core.storage import prune_layers

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="ETL: external sources -> clean tables.")
ml_app = typer.Typer(no_args_is_help=True, help="Model pipeline: clean tables -> model.")
analysis_app = typer.Typer(no_args_is_help=True, help="Analysis: layers -> tables and figures.")
app.add_typer(data_app, name="data")
app.add_typer(ml_app, name="ml")
app.add_typer(analysis_app, name="analysis")

Domain = Annotated[
    str | None,
    typer.Option(
        "--domain", "-d", help="A package under domains/; defaults to MLOPS_DOMAIN or the only one"
    ),
]


def _adapter(domain: str | None) -> DomainAdapter:
    return load_adapter(domain or Settings().domain)


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
    # Some services take their credential as a URL path segment, and the
    # safeguard lives with the client so every entry point gets it, not just this one.
    silence_request_urls()


@data_app.command()
def extract(domain: Domain = None) -> None:
    """Download every source of the domain to the raw layer.

    File sources always run. An API source runs when its credential is configured and is
    skipped out loud when it is not, so a fresh clone still builds the whole stage 1.
    """
    adapter = _adapter(domain)
    config = adapter.config
    data_dir = _data_dir(config)
    with http_client() as client:
        artifacts = extract_all(config, data_dir / "raw", client)
        api = adapter.extract(data_dir, client)
    for name, reason in api.skipped.items():
        typer.echo(f"{name}: skipped, {reason}", err=True)
    for name, artifact in (artifacts | api.artifacts).items():
        typer.echo(f"{name}: {artifact.path} ({artifact.manifest.size_bytes:,} bytes)")


@data_app.command()
def validate(domain: Domain = None) -> None:
    """Check the latest raw ingestion of every source against its contract."""
    adapter = _adapter(domain)
    config = adapter.config
    for name, source in validate_raw(adapter, _data_dir(config) / "raw").items():
        typer.echo(f"{name}: {source.frame.height:,} rows valid ({source.artifact.partition.name})")


@data_app.command()
def clean(domain: Domain = None) -> None:
    """Build the clean layer from the latest validated raw data."""
    adapter = _adapter(domain)
    config = adapter.config
    for table, path in build_clean(adapter, _data_dir(config)).items():
        typer.echo(f"{table}: {path}")


@data_app.command("run")
def data_run(domain: Domain = None) -> None:
    """Whole ETL: extract, then validate and clean."""
    extract(domain)
    clean(domain)


@ml_app.command()
def features(domain: Domain = None) -> None:
    """Build the model-ready feature table from the latest clean layer."""
    with _needs_extra("ml"):
        from mlops_core.ml.features import build_features

    adapter = _adapter(domain)
    config = adapter.config
    typer.echo(f"{config.items.features_table}: {build_features(adapter, _data_dir(config))}")


@ml_app.command()
def train(domain: Domain = None) -> None:
    """Tune and train a model, track it in MLflow, promote it if it passes the quality gate."""
    # Imported here: MLflow and LightGBM take seconds to import and no other command needs them.
    with _needs_extra("ml"):
        from mlops_core.ml.train import train_model

    adapter = _adapter(domain)
    config = adapter.config
    result = train_model(config, _data_dir(config), Settings().mlflow_tracking_uri)
    metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(result.metrics.items()))
    status = "promoted to champion" if result.promoted else "not promoted"
    typer.echo(f"{config.training.registered_model} v{result.model_version}: {status}")
    typer.echo(f"run {result.run_id}: {metrics}")


@ml_app.command()
def predict(domain: Domain = None) -> None:
    """Score the whole feature table with the champion and write the predictions."""
    with _needs_extra("ml"):
        from mlops_core.ml.predict import batch_predict

    adapter = _adapter(domain)
    config = adapter.config
    path = batch_predict(config, _data_dir(config), Settings().mlflow_tracking_uri)
    typer.echo(f"{config.items.predictions_table}: {path}")


@ml_app.command("run")
def ml_run(domain: Domain = None) -> None:
    """Whole model pipeline: features, train, then batch predictions."""
    features(domain)
    train(domain)
    predict(domain)


@analysis_app.command("run")
def analysis_run(domain: Domain = None) -> None:
    """Compute every study from the latest layers, as Parquet and CSV."""
    with _needs_extra("analysis"):
        from mlops_core.analysis.pipeline import build_analysis

    adapter = _adapter(domain)
    config = adapter.config
    # Figures are published into the repo's docs only when running from a checkout.
    docs = REPO_ROOT / "docs" / "figures"
    output = build_analysis(
        adapter,
        _data_dir(config),
        Settings().mlflow_tracking_uri,
        publish_to=docs if docs.parent.is_dir() else None,
    )
    for name, path in (output.tables | output.figures).items():
        typer.echo(f"{name}: {path}")
    for path in output.published:
        typer.echo(f"published: {path}")


@analysis_app.command()
def dashboard(domain: Domain = None, port: int = 8501) -> None:
    """Open the analysis dashboard over whatever the pipeline last wrote."""
    with _needs_extra("analysis"):
        from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).resolve().parent / "analysis" / "dashboard.py"
    # Streamlit reads the domain from the environment, like every other setting.
    os.environ["MLOPS_DOMAIN"] = _adapter(domain).config.name
    # Headless also on the command line, in case the repo's .streamlit/ is not the cwd.
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    streamlit_cli.main()


@app.command()
def secrets(domain: Domain = None) -> None:
    """Say which credentials the domain has configured, without revealing any of them."""
    configured = _adapter(domain).credentials()
    if not configured:
        typer.echo("this domain needs no credentials")
    for label, secret in configured.items():
        # Length only: enough to confirm the right value was pasted, useless if seen.
        state = f"set ({len(secret.get_secret_value())} characters)" if secret else "missing"
        typer.echo(f"{label}: {state}")


@app.command()
def sql(
    query: Annotated[str, typer.Argument(help="e.g. 'SELECT count(*) FROM clean.<table>'")],
    domain: Domain = None,
) -> None:
    """Run SQL over the latest partition of every layer."""
    with _needs_extra("data"):
        from mlops_core.catalog import connect

    adapter = _adapter(domain)
    config = adapter.config
    typer.echo(connect(_data_dir(config)).sql(query))


@app.command()
def prune(
    domain: Domain = None,
    keep: Annotated[int | None, typer.Option(help="Complete partitions to keep per table")] = None,
) -> None:
    """Delete old partitions of every layer, keeping the newest ones."""
    adapter = _adapter(domain)
    config = adapter.config
    settings = Settings()
    pruned = prune_layers(_data_dir(config), keep if keep is not None else settings.keep_partitions)
    for table, count in pruned.items():
        typer.echo(f"{table}: {count} partitions removed")
    if not pruned:
        typer.echo("nothing to prune")
