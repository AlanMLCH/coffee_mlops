# Running it in the cloud: a plan

**Status: a plan, not built.** Today the whole project runs on one laptop, and only when
someone runs it (decided 2026-09-28): no process is left collecting history on its own.
This page is what it would take to run it automated and online, cheapest step first, so
the decision can be made later with the costs in view.

## Why it cannot live on Vercel as it is

Vercel serves static pages and short-lived serverless functions. This project is a set of
long-running processes:

| Piece | What it is today | Why a serverless function cannot hold it |
|---|---|---|
| The agent's model | Ollama with `qwen3.5:4b` and `qwen3-embedding:0.6b`, about 4.3 GB of the laptop's GPU | a model server has to stay loaded in memory; functions have no GPU and do not persist |
| Retrieval | Qdrant, the vector index | a database process with its own disk |
| Tracking and registry | MLflow's server | a server with a database and an artifact store |
| Predictions | FastAPI with the champion loaded | loads models from the registry once and keeps them in memory |
| The explorer | Streamlit | a persistent server that talks to the browser over a websocket |
| Schedules | Dagster's daemon | a process whose job is to be always on |
| Data | Parquet on local disk, read through DuckDB | functions have no disk that survives the call |

The local model is the biggest reason, but not the only one: without it, the explorer and
the services still need somewhere that stays on. What does fit Vercel is a static
showcase (stage 1 below).

## Three stages, cheapest first

### 1. A static showcase - no server, no recurring cost

Publish what the pipeline already produces, as files: the figures, the studies' tables,
the explorer's layers exported as GeoJSON and JSON, the evaluation results. A static page
draws the deck.gl map in the browser from those files. No agent and no live queries: it
shows the last run, refreshed when someone runs the pipeline by hand and publishes.
Fits Vercel or GitHub Pages.

What is new: an export command (`mlops export`) that writes the layers and studies as
static files, and the static page itself.

### 2. History without a server - a scheduled workflow

The sources that need a daily read are the ones whose history is their downloads: the
ICO's page shows only the current month, the shops only today's catalogue. A scheduled
GitHub Actions workflow can run `mlops data run` once a day on the project's own runner
image and write `raw/` and `clean/` to object storage (S3, Cloudflare R2, Google Cloud
Storage). The tokens live in the repository's secrets, as CI's already do; the data still
never enters the repository.

Worth knowing before choosing it: scheduled workflows are not guaranteed to start on the
minute, and GitHub disables them in a public repository with no activity for a long time,
so the partitions by read day (`<source>_reads`) are what would show a missed day.

What is new: storage that is not a local directory (see below), and the workflow.
`MLOPS_AUTOMATE` stays off: the workflow's cron is the schedule.

### 3. The whole app online - servers, and a recurring cost

One machine, or one container service per piece, running what `docker compose` runs
today (MLflow, the API, Qdrant) plus Dagster with `MLOPS_AUTOMATE=true` and the explorer
behind a login. The open decision is the model:

| Option | For | Against |
|---|---|---|
| A GPU machine running Ollama, as today | the same model, the same evaluation | the most expensive to keep on |
| A CPU machine running Ollama | cheaper | answers go from seconds to tens of seconds |
| A hosted model API behind the same `Generator` interface | pay per question, fast | the agent was benchmarked with qwen3.5:4b: a new model has to pass the same benchmark and the same evaluation, as granite and qwen did |

The embedding model must stay the one the index was built with, or the index is rebuilt
(`make index`) and retrieval goes through its gate again.

This stage breaks the project's first rule - no cloud, no recurring costs - so it is a
decision to make, not a default.

## What the code would have to change

- **Storage.** `data_dir` is a local path everywhere. Object storage means reading and
  writing Parquet through a URL (DuckDB's `httpfs` reads it; writes need an fsspec-style
  layer in `mlops_core.storage`). To be verified when it is built, not assumed.
- **The generator.** A hosted model needs a client with the same `ask(prompt, reply)` as
  `rag.llm.LocalModel`, and has to pass `make benchmark` and `make agent-eval`.
- **A login in front of the explorer.** Streamlit has none of its own.
- **Already ready:** the Dagster definitions (switched on with `MLOPS_AUTOMATE=true`), the
  API's image, the contracts, the idempotent raw layer (an unchanged download stores
  nothing), and the log of every download (`checks.jsonl`), which is what makes an
  unattended history auditable.
