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
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import click
import polars as pl
import typer
from pydantic import SecretStr

from mlops_core.adapter import DomainAdapter, domain_dir, load_adapter
from mlops_core.config import (
    CHUNKS_TABLE,
    DOCUMENTS_TABLE,
    CorpusConfig,
    DomainConfig,
    Settings,
    env_file_names,
    unread_settings,
)
from mlops_core.data.api import silence_request_urls
from mlops_core.data.clean import build_clean
from mlops_core.data.documents import fetch_documents
from mlops_core.data.extract import extract_all, http_client
from mlops_core.data.validate import validate_raw
from mlops_core.provenance import REPO_ROOT, code_version
from mlops_core.rag.questions import (
    QUESTIONS_FILE,
    Question,
    contains,
    load_questions,
    questions_path,
    reviewed,
    save_questions,
    tally,
)
from mlops_core.storage import latest_partition, prune_layers, read_table, write_table

if TYPE_CHECKING:  # the `rag` extra's; imported where used, for installs without it
    import httpx
    from qdrant_client import QdrantClient

    from mlops_core.agent.graph import Agent
    from mlops_core.rag.llm import LocalModel
    from mlops_core.rag.providers import Generator

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
agent_app = typer.Typer(
    no_args_is_help=True, help="The agent: its tools, and the model that drives them."
)
app.add_typer(rag_app, name="rag")
app.add_typer(agent_app, name="agent")

Domain = Annotated[
    str | None,
    typer.Option(
        "--domain",
        "-d",
        help="A package under domains/, or <domain>/<subdomain>; defaults to MLOPS_DOMAIN or "
        "the only domain",
    ),
]


ModelName = Annotated[
    str | None,
    typer.Option("--model", "-m", help="One of the domain's models; defaults to all of them"),
]


Cases = Annotated[
    str | None,
    typer.Option(help="A named case set beside the domain's (evals/<name>/), e.g. holdout"),
]


def _models(config: DomainConfig, model: str | None) -> list[str]:
    """The named model, checked against the config, or every model in declared order."""
    return [config.model_named(model).name] if model else [m.name for m in config.models]


def _adapter(domain: str | None) -> DomainAdapter:
    return load_adapter(domain or Settings().domain)


def _data_dir(config: DomainConfig) -> Path:
    return Settings().data_dir / config.home


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
        files = extract_all(config, data_dir / "raw", client)
        api = adapter.extract(data_dir, client)
        corpus, absent = fetch_documents(
            config.documents,
            data_dir,
            client,
            refresh_hours=config.corpus.refresh_hours if config.corpus else None,
        )
    for name, reason in (api.skipped | absent).items():
        typer.echo(f"{name}: skipped, {reason}", err=True)
    for name, artifact in (files.artifacts | api.artifacts | corpus).items():
        typer.echo(f"{name}: {artifact.path} ({artifact.manifest.size_bytes:,} bytes)")
    if files.failed:
        for name, reason in files.failed.items():
            typer.echo(f"{name}: FAILED, {reason}", err=True)
        # The rest is stored; a source's last good download stays the one `clean` reads.
        typer.echo(
            f"{len(files.failed)} source(s) could not be downloaded; run `make extract` again "
            "later - what is already fresh is not downloaded twice.",
            err=True,
        )
        raise typer.Exit(1)


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
        status = (
            "promoted to champion"
            if result.promoted
            else f"not promoted; {result.provisional}"
            if result.provisional
            else "not promoted"
        )
        registered = config.model_named(name).training.registered_model
        typer.echo(f"{registered} v{result.model_version}: {status}")
        typer.echo(f"run {result.run_id}: {metrics}")


@ml_app.command()
def predict(domain: Domain = None, model: ModelName = None) -> None:
    """Score each model's whole feature table with its champion and write the predictions."""
    with _needs_extra("ml"):
        from mlops_core.ml.predict import batch_predict
        from mlops_core.ml.registry import NoChampion

    adapter = _adapter(domain)
    config = adapter.config
    for name in _models(config, model):
        try:
            path = batch_predict(config, name, _data_dir(config), Settings().mlflow_tracking_uri)
        except NoChampion:  # the gate never let one through: nothing to score with
            typer.echo(f"{name}: not scored - no version has passed the gate")
            continue
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


@app.command()
def explore(domain: Domain = None, port: int = 8502) -> None:
    """Open the explorer: a map of the domain's places, questions to its agent, and a
    chart of each answer.

    The map needs only the built layers; the questions need what `mlops agent ask` needs
    (Ollama, Qdrant with an index, the prediction API).
    """
    with _needs_extra("explore"):
        from streamlit.web import cli as streamlit_cli

        import mlops_core.explore.maps  # noqa: F401 - pydeck: the extra is really there
    from mlops_core.explore.style import write_theme

    app_path = Path(__file__).resolve().parent / "explore" / "app.py"
    os.environ["MLOPS_DOMAIN"] = _adapter(domain).config.tenant
    # Outside the repo: the theme is rewritten on every launch, from the code.
    theme = write_theme(Path(tempfile.gettempdir()) / "mlops-explore-theme.toml")
    sys.argv = ["streamlit", "run", str(app_path), "--server.port", str(port),
                "--server.headless", "true", "--theme.base", str(theme)]  # fmt: skip
    streamlit_cli.main()


@app.command()
def export(
    domain: Domain = None,
    to: Annotated[Path, typer.Option(help="Where the snapshot's zip is written")] = Path("dist"),
) -> None:
    """Write a snapshot of what the explorer shows - the tables its pages read, every study
    and figure - for its showcase: the same app, published without the agent
    (`MLOPS_SHOWCASE=<the zip's URL or path> mlops explore`). Tables a source's terms keep
    home are left out, as the domain's `explore.showcase` says."""
    with _needs_extra("explore"):
        from mlops_core.explore.export import export_snapshot
    config = _adapter(domain).config
    to.mkdir(parents=True, exist_ok=True)
    snapshot = export_snapshot(config, _data_dir(config), to)
    typer.echo(
        f"{snapshot.archive}: {len(snapshot.tables)} tables, {len(snapshot.filtered)} of them "
        "in part"
    )
    withheld = config.explore.showcase.withheld if config.explore else {}
    for name in snapshot.withheld:
        typer.echo(f"  withheld {name}: {withheld[name]}")


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
        drafted_by = _identified(model)
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
def index(domain: Domain = None) -> None:
    """Embed every chunk, write the vectors to Parquet, and load them into Qdrant.

    A new collection per build; the alias searches use moves onto it only once it is
    complete, and the builds before it are dropped. Parquet stays the source of truth.
    """
    with _needs_extra("rag"):
        from mlops_core.rag.llm import LocalModel, ollama_client
        from mlops_core.rag.vectors import (
            EMBEDDING_MODEL,
            EMBEDDINGS,
            EMBEDDINGS_TABLE,
            build_index,
            chunks_digest,
            embedding_table,
        )

    config = _adapter(domain).config
    _corpus(config)
    data_dir = _data_dir(config)
    chunks, _ = _corpus_tables(config)
    settings = Settings()
    client = _qdrant(settings.qdrant_url)
    with ollama_client(settings.ollama_url) as http:
        embedder = LocalModel(http, EMBEDDING_MODEL, {})
        model = _identified(embedder)
        vectors = embedder.embed(chunks["text"].to_list())
    built_at = datetime.now(UTC)
    table = embedding_table(chunks, vectors)
    source = _chunks_partition(data_dir)
    path = write_table(table, data_dir / EMBEDDINGS / EMBEDDINGS_TABLE, {CHUNKS_TABLE: source})
    collection = build_index(
        client,
        config.name,
        chunks,
        table,
        {
            "chunks_partition": source,
            "chunks_digest": chunks_digest(chunks),
            "embedding_model": model,
        },
        built_at.strftime("%Y%m%dT%H%M%SZ"),
    )
    typer.echo(f"{EMBEDDINGS_TABLE}: {path} ({table.height:,} chunks, {model})")
    typer.echo(f"qdrant: {collection}")


@rag_app.command()
def evaluate(domain: Domain = None) -> None:
    """Score every search on the ladder - BM25, dense, hybrid - and log a run for each.

    Each search after the first is judged against the ones before it, question by
    question; the per-question tables land in `evaluations/retrieval_<search>`, queryable
    with `mlops sql`, and every run says how many of the questions a person reviewed.
    """
    with _needs_extra("rag"):
        from mlops_core.rag.evaluate import GATE_METRIC, Search, evaluate_retrieval
        from mlops_core.rag.lexical import K1, B, Bm25
        from mlops_core.rag.llm import LocalModel, ollama_client
        from mlops_core.rag.vectors import (
            EMBEDDING_MODEL,
            PREFETCH,
            QUERY_TASK,
            RRF_K,
            IndexSearch,
        )

    config, corpus, path, questions = _question_set(domain)
    data_dir = _data_dir(config)
    chunks, _ = _corpus_tables(config)
    settings = Settings()
    client, built = _current_index(config, settings, chunks)

    keyword: dict[str, str | float] = {
        "k1": K1,
        "b": B,
        "stemmer": "snowball-english",
        "stop_words": "lucene-english",
    }
    semantic: dict[str, str | float] = {
        "embedding_model": built["embedding_model"],
        "query_instruction": QUERY_TASK,
    }
    with ollama_client(settings.ollama_url) as http:
        embedder = LocalModel(http, EMBEDDING_MODEL, {})

        @cache  # dense and hybrid ask for the same question's vector
        def embed(text: str) -> list[float]:
            vector: list[float] = embedder.embed([text])[0].tolist()
            return vector

        served = IndexSearch(client, config.name, chunks, embed)
        ladder: list[tuple[str, Search, dict[str, str | float]]] = [
            ("bm25", Bm25(chunks["text"].to_list()).search, keyword),
            ("dense", served.dense, semantic),
            ("hybrid", served.hybrid, keyword | semantic | {"rrf_k": RRF_K, "prefetch": PREFETCH}),
        ]
        judged: dict[str, pl.DataFrame] = {}
        for name, search, search_settings in ladder:
            run = evaluate_retrieval(
                config,
                name,
                search,
                search_settings,
                questions,
                path,
                chunks,
                corpus.chunking,
                data_dir,
                settings.mlflow_tracking_uri,
                references=judged,
            )
            judged[name] = run.frame
            scores = ", ".join(
                f"{metric} {run.overall[metric]:.3f}"
                for metric in (GATE_METRIC, "recall_at_5", "recall_at_10", "reciprocal_rank")
            )
            typer.echo(f"{name}: {scores}")
            for reference, c in run.comparisons.items():
                typer.echo(
                    f"  vs {reference}: {c.difference:+.3f} {GATE_METRIC} "
                    f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}], {c.probability_better:.0%} sure"
                )
            if run.comparisons:
                typer.echo(f"  {'passes' if run.passes else 'does not pass'} the gate")


@agent_app.command()
def benchmark(
    domain: Domain = None,
    generator: Annotated[
        list[str] | None,
        typer.Option(
            help="A model to measure - an Ollama model, or a provider's name from "
            "providers.yaml; repeat it. Defaults to the local candidates"
        ),
    ] = None,
    cases: Cases = None,
) -> None:
    """Measure how well each model writes the SQL and routes the questions.

    The bar - 70% of SQL questions right with repairs, 90% routed right - was set before
    any model was measured. Each model's verdicts land in `evaluations.agent_*` and one
    MLflow run each.
    """
    with _needs_extra("rag"):
        from mlops_core.agent.benchmark import (
            CANDIDATES,
            GENERATOR_OPTIONS,
            ROUTE_CASES_FILE,
            SQL_CASES_FILE,
            RouteCase,
            SqlCase,
            case_file,
            load_cases,
            log_benchmark,
            meets_bar,
            run_benchmark,
        )
        from mlops_core.agent.dictionary import (
            domain_dictionary,
            reads_any,
            schema_context,
            shown,
        )
        from mlops_core.agent.model_cards import ModelFinder, model_cards, served_models
        from mlops_core.agent.routing import routing_context
        from mlops_core.agent.sql import read_only, views
        from mlops_core.rag.llm import LocalModel, ollama_client
        from mlops_core.rag.providers import load_providers
        from mlops_core.rag.vectors import EMBEDDING_MODEL, QUERY_OPTIONS

    adapter = _adapter(domain)
    config = adapter.config
    home = domain_dir(config.tenant)
    data_dir = _data_dir(config)
    con = read_only(data_dir, config.parent)
    dictionary = domain_dictionary(config)
    hidden = config.agent.hidden_tables
    names = shown(views(con), hidden)
    schema = schema_context(dictionary, names)
    context = routing_context(config, dictionary, names)
    case_files = [case_file(home, SQL_CASES_FILE, cases), case_file(home, ROUTE_CASES_FILE, cases)]
    sql_cases = load_cases(case_files[0], SqlCase)
    route_cases = load_cases(case_files[1], RouteCase)
    # A question whose reference reads a table the agent is not shown is not its to answer.
    unseen = {case.question for case in sql_cases if reads_any(case.sql, hidden)}
    sql_cases = [case for case in sql_cases if case.question not in unseen]
    route_cases = [case for case in route_cases if case.question not in unseen]
    if unseen:
        typer.echo(f"{len(unseen)} questions read tables the agent is not shown: left out")
    settings = Settings()
    hosted = {provider.name for provider in load_providers(settings.providers_file)}
    for name in generator or CANDIDATES:
        with ollama_client(settings.ollama_url) as http, ExitStack() as stack:
            local = LocalModel(http, name, GENERATOR_OPTIONS)
            # A provider alone, with no fallback and no cache: the model measured is it.
            model, identity = (
                hosted_chain(settings, stack, local, only=name)
                if name in hosted
                else (local, _identified(local))
            )
            embedder = LocalModel(http, EMBEDDING_MODEL, QUERY_OPTIONS)
            linker = _linker(config, dictionary, names, embedder)
            with _api_client(settings.api_url) as api:
                served = served_models(adapter, api)
            finder = ModelFinder(model_cards(adapter, served), embedder.embed)
            sql, routes = run_benchmark(
                model,
                con,
                schema,
                context,
                sql_cases,
                route_cases,
                config.agent.sql_guards,
                linker,
                finder.listing,
            )
        summary, run_id = log_benchmark(
            config,
            identity,
            sql,
            routes,
            case_files,
            data_dir,
            settings.mlflow_tracking_uri,
            cases=cases,
        )
        verdict = "meets the bar" if meets_bar(summary) else "misses the bar"
        typer.echo(
            f"{identity}: SQL {summary['sql_accuracy']:.0%} right "
            f"({summary['sql_first_try']:.0%} at the first try), "
            f"routing {summary['route_accuracy']:.0%} right - {verdict} (run {run_id})"
        )


@agent_app.command()
def ask(
    question: Annotated[str, typer.Argument(help="A question, in English")],
    domain: Domain = None,
) -> None:
    """Answer a question with the tables, the models and the documents, citing each.

    Needs Ollama, Qdrant with a built index, and the prediction API. Every answer is one
    MLflow trace (experiment `<domain>-agent`), linked to the prompts' registry versions.
    """
    with _needs_extra("agent"):
        import mlflow

    adapter = _adapter(domain)
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{adapter.config.name}-agent")
    with agent_session(adapter, settings) as (agent, _):
        reply = agent.ask(question)
    typer.echo(reply.text)
    for source in reply.sources:
        typer.echo(source)
    if reply.sql is not None:
        typer.echo(f"sql: {reply.sql.sql}")
    if reply.prediction is not None:
        typer.echo(f"prediction ({reply.prediction.model}): {reply.prediction.request}")
    for problem in reply.problems:
        typer.echo(f"unverified: {problem}", err=True)
    typer.echo(f"route: {reply.route} | trace: {mlflow.get_last_active_trace_id()}")


@agent_app.command("providers")
def providers_command(
    check: Annotated[
        bool, typer.Option(help="Ask each provider with a key one tiny question: is the key good?")
    ] = False,
) -> None:
    """The chain of models the agent asks, in order: each hosted provider, whether its key
    is set, whether it is set aside and until when, and what it spent the last days; the
    local model is always last. Nothing is printed of a key but whether it is set."""
    with _needs_extra("rag"):
        from mlops_core.rag.providers import (
            ApiModel,
            Cooldowns,
            Ping,
            environment,
            lasting,
            load_providers,
            provider_client,
        )

    settings = Settings()
    configs = load_providers(settings.providers_file)
    env = environment()
    cooldowns = Cooldowns(settings.data_dir / "llm" / "providers.json")
    aside = cooldowns.report()["aside"]
    now = datetime.now(UTC)
    if not configs:
        typer.echo(f"{settings.providers_file}: no providers; the agent asks its local model.")
    ready = []
    for number, config in enumerate(configs, start=1):
        until = cooldowns.until(config.name, now)
        if not env.get(config.key):
            state = f"no key ({config.key} is not set)"
        elif config.paid and not settings.paid_providers:
            state = "paid: left out (MLOPS_PAID_PROVIDERS is not true)"
        elif until is not None:
            entry = aside[config.name]
            since = datetime.fromisoformat(entry["since"]) if entry.get("since") else None
            state = (
                f"set aside since {since:%Y-%m-%d %H:%M}, for {lasting(until - since)}, "
                if since is not None
                else "set aside "
            ) + f"until {until:%Y-%m-%d %H:%M} UTC: {entry['reason']}"
        else:
            state = "ready"
            ready.append(config)
        only = settings.generator == config.name
        pinned = " (the only one asked: MLOPS_GENERATOR)" if only else ""
        typer.echo(f"{number}. {config.name} - {config.model}: {state}{pinned}")
    typer.echo(f"{len(configs) + 1}. local - the Ollama model: always last, never set aside")
    for day, spent in sorted(cooldowns.report()["spent"].items())[-3:]:
        for name, usage in spent.items():
            typer.echo(f"   {day} {name}: {usage.get('calls', 0)} calls, "
                       f"{usage.get('input', 0):,} tokens in ({usage.get('cached', 0):,} cached), "
                       f"{usage.get('output', 0):,} out")  # fmt: skip
    if not check:
        return
    for config in ready:
        with provider_client(config) as client:
            model = ApiModel(client, config, SecretStr(env[config.key]))
            try:
                model.ask("Reply with ok set to true.", Ping)
            except Exception as failed:  # a check reports; it does not stop at the first
                typer.echo(f"check {config.name}: FAILED - {failed}")
            else:
                typer.echo(f"check {config.name}: answered ({model.spent.get('input', 0)} tokens)")


@agent_app.command("evaluate")
def evaluate_agent(
    domain: Domain = None,
    cases: Cases = None,
    generator: Annotated[
        str,
        typer.Option(
            help="Which model answers: local (the default, comparable with earlier runs), "
            "a provider's name from providers.yaml, or chain (the whole chain)"
        ),
    ] = "local",
    workers: Annotated[
        int,
        typer.Option(
            min=1,
            help="Questions asked at once when hosted models answer; the local model "
            "alone is asked one at a time, and takes one call at a time in a chain",
        ),
    ] = 4,
) -> None:
    """Ask the agent every routing question and check each answer end to end.

    Checks the tools that ran, verification, the query's answer against the SQL set's
    reference, the passages against the retrieval set's labels, and the prediction's
    request against the fields the question states. The answers land in
    `evaluations.agent_answers`; one MLflow run (experiment `<domain>-agent-eval`) holds
    the metrics, a trace per question, and the comparison with the previous run.
    """
    with _needs_extra("agent"):
        import mlflow

        from mlops_core.agent.benchmark import (
            GENERATOR_OPTIONS,
            ROUTE_CASES_FILE,
            SQL_CASES_FILE,
            RouteCase,
            SqlCase,
            case_digest,
            case_file,
            load_cases,
        )
        from mlops_core.agent.dictionary import reads_any
        from mlops_core.agent.evaluate import (
            known_answers,
            log_evaluation,
            record,
            run_evaluation,
            versus,
        )
        from mlops_core.agent.prompts import PROMPTS, version
        from mlops_core.rag.providers import LOCAL

    adapter = _adapter(domain)
    config = adapter.config
    home = domain_dir(config.tenant)
    case_files = [
        case_file(home, file, cases) for file in (ROUTE_CASES_FILE, SQL_CASES_FILE, QUESTIONS_FILE)
    ]
    truths = known_answers(
        load_cases(case_files[0], RouteCase),
        load_cases(case_files[1], SqlCase),
        _retrieval_questions(config, case_files[2]),
    )
    hidden = config.agent.hidden_tables
    unseen = [t for t in truths if t.sql is not None and reads_any(t.sql, hidden)]
    truths = [t for t in truths if t not in unseen]
    if unseen:
        typer.echo(f"{len(unseen)} questions read tables the agent is not shown: left out")
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{config.name}-agent-eval")
    # Never from the reply cache: an evaluation measures the model, not what it said before.
    only = None if generator == "chain" else generator
    with (
        agent_session(adapter, settings, only, cache=False) as (agent, identity),
        mlflow.start_run() as run,
    ):
        # Hosted models wait on the network: several questions at once. The local model
        # alone, one at a time - it takes its calls one at a time anyway (`LOCAL_TURN`).
        at_once = 1 if generator == LOCAL else workers
        answers = run_evaluation(agent.ask, truths, agent.con, workers=at_once)
        table, previous = record(answers, _data_dir(config), cases=cases)
        comparison = versus(previous, answers) if previous is not None else None
        version_ = code_version()
        mlflow.set_tags(version_.as_tags() if version_ else {})
        summary = log_evaluation(
            answers,
            comparison,
            {
                "generator": identity,
                "questions_at_once": at_once,
                **{f"option_{k}": v for k, v in GENERATOR_OPTIONS.items()},
                **{
                    f"prompt_{name}": version(template, reply) if reply else "domain"
                    for name, (template, reply) in PROMPTS.items()
                },
                "cases": answers.height,
                "cases_written_by": "assistant",
                "case_set": cases or "default",
                "case_files_sha256": case_digest(case_files),
            },
            table,
        )
    typer.echo(
        f"{summary['correct']:.0%} correct, {summary['verified']:.0%} verified, "
        f"routing {summary['route_accuracy']:.0%}; median {summary['seconds_median']:.0f} s "
        f"a question (run {run.info.run_id})"
    )
    for name in ("sql", "passage", "item"):
        if name in summary:
            typer.echo(f"  {name}: {summary[name]:.0%} of the questions it applies to")
    for row in answers.filter(~pl.col("correct")).iter_rows(named=True):
        why = [
            f"route {row['route']}" if not row["route_ok"] else "",
            f"tools {row['tools'] or 'none'}" if not row["tools_ok"] else "",
            "wrong query result" if row["sql_ok"] is False else "",
            "no relevant passage" if row["passage_ok"] is False else "",
            f"item: {row['item_errors']}" if row["item_ok"] is False else "",
            f"unverified: {row['problems']}" if not row["verified"] else "",
        ]
        typer.echo(f"  x {row['case_id']}: {'; '.join(w for w in why if w)}")
    if comparison is not None:
        typer.echo(
            f"vs the previous run: {comparison.difference:+.0%} correct "
            f"[{comparison.ci_low:+.0%}, {comparison.ci_high:+.0%}], "
            f"{comparison.probability_better:.0%} sure it is better"
        )


@app.command()
def monitor(
    domain: Domain = None,
    model: ModelName = None,
    retrain: Annotated[
        bool, typer.Option(help="Retrain the models that call for it; the gate decides")
    ] = False,
) -> None:
    """Compare each model's newest period with the earlier ones, and say which should be
    retrained: drifted features, a drifted target, or an error above the interval the
    champion was accepted with.

    Each model's comparison lands in `monitoring.<model>_drift`, with Evidently's report
    beside it, and in an MLflow run (experiment `<domain>-monitoring`). With --retrain,
    the models due are rebuilt - features, training, batch scores - and the gate decides
    whether the candidate is served.
    """
    with _needs_extra("monitoring"):
        from mlops_core.monitoring.drift import monitor_model

    config = _adapter(domain).config
    settings = Settings()
    due: list[str] = []
    for name in _models(config, model):
        result = monitor_model(config, name, _data_dir(config), settings.mlflow_tracking_uri)
        if result is None:
            typer.echo(f"{name}: one period only, nothing to compare it with yet")
            continue
        typer.echo(
            f"{name}: {result.current} against {', '.join(result.reference)} - "
            f"{result.drifted_share:.0%} of the features drifted"
        )
        if result.trained_run is not None:
            # A source that stopped changing keeps its drift: retraining on the same rows
            # again would give the same candidate, and the gate the same answer.
            typer.echo(f"  the same rows run {result.trained_run} learned from: no retraining due")
            for reason in result.reasons:
                typer.echo(f"  drift, recorded: {reason}")
            continue
        for reason in result.reasons:
            typer.echo(f"  due for retraining: {reason}")
        if result.retrain:
            due.append(name)
        else:
            typer.echo("  no reason to retrain")
    if not retrain:
        return
    for name in due:
        typer.echo(f"retraining {name}")
        ml_run(domain, name)


@app.command("mcp")
def mcp_server(
    domain: Domain = None,
    over_http: Annotated[
        bool,
        typer.Option(
            "--http",
            help="Serve over streamable HTTP on 127.0.0.1 instead of stdio, for a client that "
            "connects to a running server rather than starting one",
        ),
    ] = False,
    port: Annotated[int, typer.Option(help="The HTTP port, with --http")] = 8765,
) -> None:
    """Serve the agent's tools over MCP, on stdio: for Claude Desktop, Claude Code or an IDE.

    The same tools the agent uses - locked-down SQL, one prediction per model, document
    search - plus the explorer's segments and map layers, a chart of any query, and the
    data dictionary, findings, studies, model cards and tables' freshness as resources.
    Needs Ollama, Qdrant with a built index, and the prediction API; on stdio, stdout is
    the protocol, so everything else goes to stderr.
    """
    with _needs_extra("mcp"):
        from mlops_core.agent.dictionary import domain_dictionary, schema_context
        from mlops_core.agent.mcp_server import build_server
        from mlops_core.agent.sql import read_only, views
        from mlops_core.agent.tools import cite
        from mlops_core.explore.layers import areas_if_built
        from mlops_core.rag.llm import LocalModel, ollama_client
        from mlops_core.rag.vectors import EMBEDDING_MODEL, QUERY_OPTIONS

    adapter = _adapter(domain)
    config = adapter.config
    settings = Settings()
    con = read_only(_data_dir(config), config.parent)
    areas = areas_if_built(con, config.explore)
    dictionary = domain_dictionary(config)
    with ollama_client(settings.ollama_url) as http, _api_client(settings.api_url) as api:
        embedder = LocalModel(http, EMBEDDING_MODEL, QUERY_OPTIONS)
        passages, titles = _documents(config, settings, embedder)
        server = build_server(
            adapter,
            con,
            schema_context(dictionary, views(con)),
            passages,
            api,
            lambda passage: cite(passage, titles),
            areas,
            _data_dir(config),
            settings.explore_url,
        )
        if over_http:
            # Bound to this machine only: the tools read the data, and nothing here asks
            # who is calling.
            server.run("streamable-http", host="127.0.0.1", port=port)
        else:
            server.run("stdio")


@contextmanager
def agent_session(
    adapter: DomainAdapter,
    settings: Settings,
    generator: str | None = None,
    cache: bool | None = None,
) -> Iterator[tuple["Agent", str]]:
    """The agent with every service it needs - Ollama, the index, the prediction API -
    and its generator's identity. The generator is the chain of hosted models whose keys
    are set, then the local one (`rag.providers`); `generator` (else the setting) asks
    only one of them. `cache` (else the setting) keeps replies by prompt. Tracking must
    already point at MLflow: the prompts are registered there. Public: the explorer app
    asks this same agent."""
    from mlops_core.agent.benchmark import GENERATOR_OPTIONS
    from mlops_core.agent.dictionary import domain_dictionary, offered, schema_context, shown
    from mlops_core.agent.graph import AGENT_GENERATOR, Agent
    from mlops_core.agent.model_cards import ModelFinder, model_cards, served_models
    from mlops_core.agent.registry import register_prompts
    from mlops_core.agent.routing import routing_context
    from mlops_core.agent.sql import read_only, views
    from mlops_core.agent.study_cards import StudyFinder, study_cards
    from mlops_core.agent.text_to_sql import VOTE_SEED, Generator
    from mlops_core.rag.llm import LocalModel, ollama_client
    from mlops_core.rag.vectors import EMBEDDING_MODEL, QUERY_OPTIONS

    config = adapter.config
    con = read_only(_data_dir(config), config.parent)
    dictionary = domain_dictionary(config)
    prompts = register_prompts(settings.mlflow_tracking_uri)
    with (
        ollama_client(settings.ollama_url) as http,
        _api_client(settings.api_url) as api,
        ExitStack() as stack,
    ):
        local = LocalModel(http, AGENT_GENERATOR, GENERATOR_OPTIONS)
        chosen, identity = hosted_chain(
            settings,
            stack,
            local,
            generator if generator is not None else settings.generator,
            settings.reply_cache if cache is None else cache,
        )
        embedder = LocalModel(http, EMBEDDING_MODEL, QUERY_OPTIONS)
        passages, titles = _documents(config, settings, embedder)
        names = shown(views(con), config.agent.hidden_tables)
        linker = _linker(config, dictionary, names, embedder)
        # The models the API serves, each a card: a question is shown the closest.
        served = served_models(adapter, api)
        finder = ModelFinder(model_cards(adapter, served), embedder.embed)
        # The studies, each a card too, when the domain shows them.
        studies = (
            StudyFinder(
                study_cards(offered(dictionary, names)), embedder.embed, config.agent.study_cards
            )
            if config.agent.study_cards
            else None
        )
        # The votes are the local model's, sampled: whichever model answers first.
        voters: list[Generator] = [
            LocalModel(http, AGENT_GENERATOR,
                       GENERATOR_OPTIONS | {"temperature": config.agent.vote_temperature,
                                            "seed": VOTE_SEED + n})
            for n in range(config.agent.sql_votes - 1)
        ]  # fmt: skip
        yield (
            Agent(
                chosen,
                adapter,
                con,
                schema_context(dictionary, names),
                routing_context(config, dictionary, names, served),
                passages,
                api,
                titles,
                prompts,
                list(config.agent.sql_guards),
                linker,
                voters,
                finder,
                library=config.corpus is not None,
                studies=studies,
                check_result=config.agent.check_result,
            ),
            identity,
        )


def hosted_chain(
    settings: Settings,
    stack: ExitStack,
    local: "LocalModel",
    only: str | None = None,
    cache: bool = False,
) -> tuple["Generator", str]:
    """The generator the agent asks, and its identity: the hosted models whose keys are
    set, in `providers.yaml`'s order (paid ones only if the setting allows), then the local
    model. `only` asks just one: "local", or a provider's name - then there is no fallback,
    so a measurement of one model is of that model. Public: the benchmark uses it too."""
    from mlops_core.rag.providers import (
        LOCAL,
        ApiModel,
        CachedGenerator,
        Chain,
        Cooldowns,
        ReplyCache,
        environment,
        load_providers,
        provider_client,
        usable,
    )

    configs = load_providers(settings.providers_file)
    found = usable(configs, environment(), settings.paid_providers, only)
    members = [
        ApiModel(stack.enter_context(provider_client(config)), config, key) for config, key in found
    ]
    local_identity = _identified(local) if only in (None, LOCAL) else None
    generator: Generator = local
    if members:
        state = settings.data_dir / "llm" / "providers.json"
        chain = Chain(members, local if local_identity else None, Cooldowns(state),
                      local_name=local_identity or LOCAL)  # fmt: skip
        generator = chain
    if cache:
        replies = ReplyCache(settings.data_dir / "llm" / "replies.sqlite")
        stack.callback(replies.close)
        generator = CachedGenerator(generator, replies)
    names = [member.model for member in members] + ([local_identity] if local_identity else [])
    return generator, " > ".join(names)


def _linker(
    config: DomainConfig, dictionary: str, names: set[str], embedder: "LocalModel"
) -> "Callable[[str], str] | None":
    """The sections a question needs, when the domain asks for linking; else None, and
    the SQL writer sees the whole dictionary."""
    from mlops_core.agent.dictionary import SchemaLinker, offered
    from mlops_core.rag.vectors import query_text

    k = config.agent.schema_sections
    if k is None:
        return None
    return SchemaLinker(offered(dictionary, names), embedder.embed, k, query_text)


def _documents(
    config: DomainConfig, settings: Settings, embedder: "LocalModel"
) -> tuple[Callable[[str, int], list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """How the agent searches the domain's documents, and each document's citation. A
    domain with no corpus - a business that brings its tables, not a library - has none:
    its agent answers from its tables and models, and finds no passage to cite."""
    from mlops_core.rag.vectors import IndexSearch

    if config.corpus is None:
        return (lambda question, k: []), {}
    chunks, documents = _corpus_tables(config)
    client, _ = _current_index(config, settings, chunks)
    search = IndexSearch(
        client, config.name, chunks, lambda text: embedder.embed([text])[0].tolist()
    )
    return search.passages, {row["document_id"]: row for row in documents.iter_rows(named=True)}


def _current_index(
    config: DomainConfig, settings: Settings, chunks: pl.DataFrame
) -> tuple["QdrantClient", dict[str, Any]]:
    """The index, if it was built from the chunks on disk - the same ids and text, whatever
    partition they sit in now; a plain word if not."""
    from mlops_core.rag.vectors import chunks_digest, index_metadata

    client = _qdrant(settings.qdrant_url)
    built = index_metadata(client, config.name)
    if built.get("chunks_digest") != chunks_digest(chunks):
        typer.echo("The index was built from other chunks: rebuild it with `make index`", err=True)
        raise typer.Exit(code=1)
    return client, built


def _api_client(url: str) -> "httpx.Client":
    """The prediction API, as the agent's prediction tool reaches it."""
    import httpx

    return httpx.Client(base_url=url, timeout=60.0)


def _identified(model: "LocalModel") -> str:
    """model@digest, or a plain word on why the model cannot be used."""
    try:
        return f"{model.model}@{model.digest()}"
    except (ConnectionError, LookupError) as unavailable:
        typer.echo(str(unavailable), err=True)
        raise typer.Exit(code=1) from unavailable


def _qdrant(url: str) -> "QdrantClient":
    """A client for the index, or a plain word on how to start it."""
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import ResponseHandlingException

    client = QdrantClient(url=url)
    try:
        client.get_collections()
    except ResponseHandlingException as down:
        typer.echo(f"Qdrant is not answering at {url}: docker compose --profile ai up -d", err=True)
        raise typer.Exit(code=1) from down
    return client


def _chunks_partition(data_dir: Path) -> str:
    partition = latest_partition(data_dir / "clean" / CHUNKS_TABLE)
    return partition.name if partition else ""


def _retrieval_questions(config: DomainConfig, path: Path) -> list[Question]:
    """The questions a passage answers; none for a domain with no documents, whose agent
    is judged on its tables and models alone."""
    return load_questions(path, config.corpus.topics) if config.corpus else []


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
    path = questions_path(domain_dir(config.tenant))
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
    """Say which credentials the domain has configured, without revealing any of them, and
    which variables look like settings but are read by nothing."""
    adapter = _adapter(domain)
    configured = adapter.credentials()
    if not configured:
        typer.echo("this domain needs no credentials")
    for label, secret in configured.items():
        # Length only: enough to confirm the right value was pasted, useless if seen.
        state = f"set ({len(secret.get_secret_value())} characters)" if secret else "missing"
        typer.echo(f"{label}: {state}")
    names = {*env_file_names(Path(".env")), *os.environ}
    for name, why in sorted(unread_settings(names, adapter.config.name).items()):
        typer.echo(f"{name}: {why}", err=True)


STATES = {True: "[ok]   ", False: "[to do]", None: "[note] "}


@app.command()
def status(domain: Domain = None) -> None:
    """What is ready and what is left to do - keys, data, models, services - each with the
    command that makes it ready."""
    with _needs_extra("data"):
        from mlops_core import status as checks

    adapter = _adapter(domain)
    config, settings = adapter.config, Settings()
    data_dir = _data_dir(config)
    findings = [*checks.key_findings(adapter), *checks.data_findings(adapter, data_dir)]
    with http_client() as http:
        services = checks.service_findings(settings, config, data_dir, http)
    registry_up = next(f.ready for f in services if f.name == "MLflow")
    if registry_up:
        findings += checks.model_findings(config, data_dir, _champion_version(settings))
    else:
        findings.append(checks.Finding(checks.MODELS, "champions", None, "unknown: MLflow is down"))
    findings += services
    for section, found in checks.by_section(findings).items():
        typer.echo(section)
        for finding in found:
            fix = f"  ->  {finding.fix}" if finding.fix else ""
            typer.echo(f"  {STATES[finding.ready]} {finding.name}: {finding.detail}{fix}")
    commands = checks.to_do(findings)
    typer.echo("\nNothing to do." if not commands else "\nTo do, in this order:")
    for command, fixes in commands.items():
        typer.echo(f"  {command}   ({', '.join(fixes)})")


def _champion_version(settings: Settings) -> Any:
    """A lookup of each registered model's champion version, None when there is none."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    def champion(registered_model: str) -> str | None:
        try:
            return str(client.get_model_version_by_alias(registered_model, "champion").version)
        except MlflowException:  # no alias, or no such model: the gate never promoted one
            return None

    return champion


SQL_ROWS = 1_000  # `mlops sql` is read by a person: more than the agent's, still bounded


@app.command()
def sql(
    query: Annotated[str, typer.Argument(help="e.g. 'SELECT count(*) FROM clean.<table>'")],
    domain: Domain = None,
) -> None:
    """Run a SELECT over the latest partition of every layer - in the session the domain's
    agent gets: its own layers and, for a subdomain, the parent tables it lists, and no
    other tenant's files, whoever asks."""
    with _needs_extra("data"):
        from mlops_core.agent.sql import read_only, run_select

    adapter = _adapter(domain)
    config = adapter.config
    result = run_select(read_only(_data_dir(config), config.parent), query, max_rows=SQL_ROWS)
    typer.echo(result.as_text())


@app.command()
def prune(
    domain: Domain = None,
    keep: Annotated[int | None, typer.Option(help="Complete partitions to keep per table")] = None,
) -> None:
    """Delete old builds of the derived layers, keeping the newest ones. The raw layer is
    never pruned: every download stays, as the record the rest is rebuilt from."""
    config = _adapter(domain).config
    settings = Settings()
    pruned = prune_layers(_data_dir(config), keep if keep is not None else settings.keep_partitions)
    for table, count in pruned.items():
        typer.echo(f"{table}: {count} partitions removed")
    if not pruned:
        typer.echo("nothing to prune")
