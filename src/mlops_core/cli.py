"""Command-line entry point.

Two independent pipelines, one command group each. `data` produces the canonical
clean tables; `ml` consumes them from disk. `rag` holds the questions retrieval is
judged by: drafted by a local model, decided by a person. Every step runs on its own;
`run` chains the steps of one pipeline when that is what you want. Every command takes
`--domain`: the CLI knows the pipeline, the domain's adapter knows the rest. The `ml`
steps also take `--model`; without it they run every model the domain declares.
"""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

import click
import polars as pl
import typer

from mlops_core.adapter import DomainAdapter, domain_dir, load_adapter
from mlops_core.config import CHUNKS_TABLE, DOCUMENTS_TABLE, CorpusConfig, DomainConfig, Settings
from mlops_core.data.api import silence_request_urls
from mlops_core.data.clean import build_clean
from mlops_core.data.documents import fetch_documents
from mlops_core.data.extract import extract_all, http_client
from mlops_core.data.validate import validate_raw
from mlops_core.provenance import REPO_ROOT
from mlops_core.rag.questions import (
    Question,
    contains,
    load_questions,
    questions_path,
    reviewed,
    save_questions,
    tally,
)
from mlops_core.storage import prune_layers, read_table

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="ETL: external sources -> clean tables.")
ml_app = typer.Typer(no_args_is_help=True, help="Model pipeline: clean tables -> model.")
analysis_app = typer.Typer(no_args_is_help=True, help="Analysis: layers -> tables and figures.")
rag_app = typer.Typer(
    no_args_is_help=True, help="RAG: the questions retrieval is judged by, reviewed by a person."
)
app.add_typer(data_app, name="data")
app.add_typer(ml_app, name="ml")
app.add_typer(analysis_app, name="analysis")
app.add_typer(rag_app, name="rag")

Domain = Annotated[
    str | None,
    typer.Option(
        "--domain", "-d", help="A package under domains/; defaults to MLOPS_DOMAIN or the only one"
    ),
]


ModelName = Annotated[
    str | None,
    typer.Option("--model", "-m", help="One of the domain's models; defaults to all of them"),
]


def _models(config: DomainConfig, model: str | None) -> list[str]:
    """The named model, checked against the config, or every model in declared order."""
    return [config.model_named(model).name] if model else [m.name for m in config.models]


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
    skipped out loud when it is not, so a fresh clone still builds the whole stage 1. A
    document a publisher refuses to serve is skipped the same way, saying where to put it.
    """
    adapter = _adapter(domain)
    config = adapter.config
    data_dir = _data_dir(config)
    with http_client() as client:
        artifacts = extract_all(config, data_dir / "raw", client)
        api = adapter.extract(data_dir, client)
        corpus, absent = fetch_documents(config.documents, data_dir, client)
    for name, reason in (api.skipped | absent).items():
        typer.echo(f"{name}: skipped, {reason}", err=True)
    for name, artifact in (artifacts | api.artifacts | corpus).items():
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
def features(domain: Domain = None, model: ModelName = None) -> None:
    """Build each model's feature table from the latest clean layer."""
    with _needs_extra("ml"):
        from mlops_core.ml.features import build_features

    adapter = _adapter(domain)
    config = adapter.config
    for name in _models(config, model):
        path = build_features(adapter, name, _data_dir(config))
        typer.echo(f"{config.model_named(name).features_table}: {path}")


@ml_app.command()
def train(domain: Domain = None, model: ModelName = None) -> None:
    """Tune and train each model, track it in MLflow, promote it if it passes the gate."""
    # Imported here: MLflow and LightGBM take seconds to import and no other command needs them.
    with _needs_extra("ml"):
        from mlops_core.ml.train import train_model

    adapter = _adapter(domain)
    config = adapter.config
    for name in _models(config, model):
        result = train_model(config, name, _data_dir(config), Settings().mlflow_tracking_uri)
        metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(result.metrics.items()))
        status = "promoted to champion" if result.promoted else "not promoted"
        registered = config.model_named(name).training.registered_model
        typer.echo(f"{registered} v{result.model_version}: {status}")
        typer.echo(f"run {result.run_id}: {metrics}")


@ml_app.command()
def predict(domain: Domain = None, model: ModelName = None) -> None:
    """Score each model's whole feature table with its champion and write the predictions."""
    with _needs_extra("ml"):
        from mlops_core.ml.predict import batch_predict

    adapter = _adapter(domain)
    config = adapter.config
    for name in _models(config, model):
        path = batch_predict(config, name, _data_dir(config), Settings().mlflow_tracking_uri)
        typer.echo(f"{config.model_named(name).predictions_table}: {path}")


@ml_app.command("run")
def ml_run(domain: Domain = None, model: ModelName = None) -> None:
    """Whole model pipeline: features, train, then batch predictions."""
    features(domain, model)
    train(domain, model)
    predict(domain, model)


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


@rag_app.command()
def draft(
    domain: Domain = None,
    per_topic: Annotated[
        int, typer.Option(min=1, help="Questions nobody rejected that each topic should have")
    ] = 12,
    drafter: Annotated[
        str | None, typer.Option(help="The Ollama model that drafts; defaults to the chosen one")
    ] = None,
) -> None:
    """Have the local model draft retrieval questions until every topic has enough.

    Each draft is saved as it is written, so an interrupted run loses nothing and the
    next one carries on where it stopped. Drafts wait for `mlops rag review`.
    """
    with _needs_extra("rag"):
        from mlops_core.rag.llm import LocalModel, ollama_client
    from mlops_core.rag.questions import DRAFTING_MODEL, OPTIONS, Draft, draft_questions

    config, corpus, path, questions = _question_set(domain)
    chunks, documents = _corpus_tables(config)
    with ollama_client(Settings().ollama_url) as client:
        model = LocalModel(client, drafter or DRAFTING_MODEL, OPTIONS)
        try:
            drafted_by = f"{model.model}@{model.digest()}"
        except (ConnectionError, LookupError) as unavailable:
            typer.echo(str(unavailable), err=True)
            raise typer.Exit(code=1) from unavailable
        drafts = draft_questions(
            chunks,
            documents,
            corpus.topics,
            questions,
            per_topic,
            lambda prompt: model.ask(prompt, Draft),
            drafted_by,
            date.today(),
        )
        for question in drafts:
            questions.append(question)
            save_questions(path, questions)
            typer.echo(f"{question.id}: {question.question}")
    _echo_tally(questions)


@rag_app.command()
def review(domain: Domain = None) -> None:
    """Go through the drafts one at a time: accept, edit, reject, skip or quit.

    Every decision is saved when it is made, so a review can stop at any question and
    resume there. Run it in a terminal of your own: it waits for your keys.
    """
    config, _, path, questions = _question_set(domain)
    chunks, documents = _corpus_tables(config)
    for question in [q for q in questions if q.status == "draft"]:
        _show(question, chunks, documents)
        choice = typer.prompt(
            "[a]ccept [e]dit [r]eject [s]kip [q]uit",
            type=click.Choice(["a", "e", "r", "s", "q"]),
            show_choices=False,
        )
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "e":
            text = typer.prompt("Question", default=question.question)
            answer = typer.prompt("Answer", default=question.answer)
            decided = reviewed(question, "accept", date.today(), wording=text, answer=answer)
        else:
            decided = reviewed(question, "accept" if choice == "a" else "reject", date.today())
        questions = [decided if q.id == decided.id else q for q in questions]
        save_questions(path, questions)
    _echo_tally(questions)


@rag_app.command()
def evaluate(domain: Domain = None) -> None:
    """Search every question with BM25, score the rankings and log the run to MLflow.

    The per-question table lands in `evaluations/retrieval_bm25`, queryable with
    `mlops sql`; the run says how many of the questions a person reviewed.
    """
    with _needs_extra("rag"):
        from mlops_core.rag.evaluate import evaluate_retrieval
        from mlops_core.rag.lexical import K1, B, Bm25

    config, corpus, path, questions = _question_set(domain)
    chunks, _ = _corpus_tables(config)
    index = Bm25(chunks["text"].to_list())
    run = evaluate_retrieval(
        config,
        "bm25",
        index.search,
        {"k1": K1, "b": B, "stemmer": "snowball-english", "stop_words": "lucene-english"},
        questions,
        path,
        chunks,
        corpus.chunking,
        _data_dir(config),
        Settings().mlflow_tracking_uri,
    )
    for name, value in sorted(run.overall.items()):
        typer.echo(f"{name}: {value:.3f}")
    typer.echo(f"table: {run.table}")
    typer.echo(f"run {run.run_id}")


def _corpus(config: DomainConfig) -> CorpusConfig:
    if config.corpus is None:
        typer.echo(f"{config.name} has no corpus: nothing to ask questions about", err=True)
        raise typer.Exit(code=1)
    return config.corpus


def _question_set(
    domain: str | None,
) -> tuple[DomainConfig, CorpusConfig, Path, list[Question]]:
    """The domain's config and corpus, where its question set lives, and the set as it
    stands."""
    config = _adapter(domain).config
    corpus = _corpus(config)
    path = questions_path(domain_dir(config.name))
    return config, corpus, path, load_questions(path, corpus.topics)


def _corpus_tables(config: DomainConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The latest chunks and documents: what questions are drafted from and shown with."""
    clean_dir = _data_dir(config) / "clean"
    return read_table(clean_dir / CHUNKS_TABLE), read_table(clean_dir / DOCUMENTS_TABLE)


def _show(question: Question, chunks: pl.DataFrame, documents: pl.DataFrame) -> None:
    """What a reviewer needs to judge a draft: where it came from, the passage, the draft."""
    source = question.source
    document = documents.filter(pl.col("document_id") == source.document_id).row(0, named=True)
    passage = chunks.filter(pl.col("chunk_id") == question.source_chunk)
    if passage.is_empty():  # the corpus was cut again since: find the excerpt instead
        mine = chunks.filter(pl.col("document_id") == source.document_id)
        passage = mine.filter(
            pl.col("text").map_elements(
                lambda text: contains(text, source.excerpt), return_dtype=pl.Boolean
            )
        )
    title = passage["part_title"].drop_nulls()
    where = f"section '{title[0]}'" if len(title) else f"page {source.part}"
    typer.echo("")
    typer.echo(f"=== {question.id} ({question.topic})")
    typer.echo(f'{document["publisher"]}, "{document["title"]}", {where}')
    typer.echo("")
    for text in passage["text"]:
        typer.echo(text)
    typer.echo("")
    typer.echo(f"Q: {question.question}")
    typer.echo(f"A: {question.answer}")
    typer.echo(f"Excerpt (the label): {source.excerpt}")


def _echo_tally(questions: list[Question]) -> None:
    for topic, counts in sorted(tally(questions).items()):
        statuses = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        typer.echo(f"{topic}: {statuses}")


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
