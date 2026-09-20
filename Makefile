DOMAIN ?= coffee
# Compose profile to start: ml | api | ai | all (repeat with PROFILE="ml --profile api")
PROFILE ?= ml

.PHONY: help install lint format typecheck test test-network check data extract validate clean-layer ml features train predict sql services-up services-down

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
	@echo "clean-layer  build the clean layer (coffee_reviews, market_context)"
	@echo "ml         whole model pipeline: features + train"
	@echo "features   build the model-ready feature table"
	@echo "train      tune, train, track in MLflow and promote if it passes the quality gate"
	@echo "predict    score the feature table with the champion (batch)"
	@echo "services-up / services-down  start / stop services (PROFILE=$(PROFILE))"
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
	uv run pytest --cov=coffee_mlops --cov-report=term-missing

test-network:
	uv run pytest -m network

check: lint typecheck test

data:
	uv run coffee-mlops data run --domain $(DOMAIN)

extract:
	uv run coffee-mlops data extract --domain $(DOMAIN)

validate:
	uv run coffee-mlops data validate --domain $(DOMAIN)

clean-layer:
	uv run coffee-mlops data clean --domain $(DOMAIN)

ml:
	uv run coffee-mlops ml run --domain $(DOMAIN)

features:
	uv run coffee-mlops ml features --domain $(DOMAIN)

sql:
	uv run coffee-mlops sql "$(Q)" --domain $(DOMAIN)

train:
	uv run coffee-mlops ml train --domain $(DOMAIN)

predict:
	uv run coffee-mlops ml predict --domain $(DOMAIN)

services-up:
	docker compose --profile $(PROFILE) up -d --wait

services-down:
	docker compose down
