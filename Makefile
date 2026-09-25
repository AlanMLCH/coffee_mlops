DOMAIN ?= coffee
KEEP ?= 3
# Dagster keeps its run history here instead of a throwaway temp dir.
export DAGSTER_HOME := $(CURDIR)/.dagster

# Compose profile to start: ml | api | ai | all (repeat with PROFILE="ml --profile api")
PROFILE ?= ml

.PHONY: help install lint format typecheck test test-network check data extract validate clean-layer ml features train predict analysis dashboard questions review index retrieval benchmark sql prune dagster services-up services-down

help:
	@echo "install    venv + dependencies + git hooks"
	@echo "lint       ruff lint and format check"
	@echo "format     apply ruff fixes and formatting"
	@echo "typecheck  mypy --strict"
	@echo "test       pytest (offline tests only)"
	@echo "check      lint + typecheck + test (what CI runs)"
	@echo "test-network  check upstream URLs still answer (hits the internet)"
	@echo "data       whole ETL: extract + validate + clean (DOMAIN=$(DOMAIN))"
	@echo "extract    download DOMAIN sources to the raw layer"
	@echo "validate   check the latest raw ingestion against its Pandera contracts"
	@echo "clean-layer  build the clean layer (every clean table of DOMAIN)"
	@echo "ml         whole model pipeline: features + train + predict (every model, or MODEL=<name>)"
	@echo "features   build the model-ready feature table"
	@echo "train      tune, train, track in MLflow and promote if it passes the quality gate"
	@echo "predict    score the feature table with the champion (batch)"
	@echo "dagster    open the asset graph UI (http://localhost:3000)"
	@echo "services-up / services-down  start / stop services (PROFILE=$(PROFILE))"
	@echo "prune      drop old partitions, keeping the newest (KEEP=$(KEEP))"
	@echo "analysis   compute the studies (tables + figures) from the latest layers"
	@echo "dashboard  open the analysis dashboard (http://localhost:8501)"
	@echo "questions  have the local model draft retrieval questions (needs Ollama)"
	@echo "review     accept, edit or reject the drafts, one at a time (in your own terminal)"
	@echo "index      embed the chunks and (re)build the Qdrant index (needs Ollama + PROFILE=ai)"
	@echo "retrieval  score BM25, dense and hybrid search on the question set, gated, in MLflow"
	@echo "benchmark  measure the local models that could drive the agent (SQL, routing)"
	@echo "sql        query any layer, e.g. make sql Q=\"SELECT count(*) FROM clean.coffee_reviews\""

install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=mlops_core --cov=domains --cov-report=term-missing

test-network:
	uv run pytest -m network

check: lint typecheck test

data:
	uv run mlops data run --domain $(DOMAIN)

extract:
	uv run mlops data extract --domain $(DOMAIN)

validate:
	uv run mlops data validate --domain $(DOMAIN)

clean-layer:
	uv run mlops data clean --domain $(DOMAIN)

ml:
	uv run mlops ml run --domain $(DOMAIN) $(if $(MODEL),--model $(MODEL))

features:
	uv run mlops ml features --domain $(DOMAIN) $(if $(MODEL),--model $(MODEL))

sql:
	uv run mlops sql "$(Q)" --domain $(DOMAIN)

train:
	uv run mlops ml train --domain $(DOMAIN) $(if $(MODEL),--model $(MODEL))

predict:
	uv run mlops ml predict --domain $(DOMAIN) $(if $(MODEL),--model $(MODEL))

services-up:
	docker compose --profile $(PROFILE) up -d --wait

services-down:
	docker compose down

dagster:
	uv run python -c "import pathlib; pathlib.Path('.dagster').mkdir(exist_ok=True)"
	uv run dagster dev -m mlops_core.orchestration.definitions

prune:
	uv run mlops prune --domain $(DOMAIN) --keep $(KEEP)

analysis:
	uv run mlops analysis run --domain $(DOMAIN)

dashboard:
	uv run mlops analysis dashboard --domain $(DOMAIN)

questions:
	uv run mlops rag draft --domain $(DOMAIN)

review:
	uv run mlops rag review --domain $(DOMAIN)

index:
	uv run mlops rag index --domain $(DOMAIN)

retrieval:
	uv run mlops rag evaluate --domain $(DOMAIN)

benchmark:
	uv run mlops agent benchmark --domain $(DOMAIN)
