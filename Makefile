DOMAIN ?= coffee

.PHONY: help install lint format typecheck test test-network check extract validate clean-layer features sql

help:
	@echo "install    venv + dependencies + git hooks"
	@echo "lint       ruff lint and format check"
	@echo "format     apply ruff fixes and formatting"
	@echo "typecheck  mypy --strict"
	@echo "test       pytest (offline tests only)"
	@echo "check      lint + typecheck + test (what CI runs)"
	@echo "test-network  check upstream URLs still answer (hits the internet)"
	@echo "extract    download DOMAIN sources to the raw layer (DOMAIN=$(DOMAIN))"
	@echo "validate   check the latest raw ingestion against its Pandera contracts"
	@echo "clean-layer  build the clean layer (coffee_reviews, market_context)"
	@echo "features   build the model-ready feature table"
	@echo "sql        query any layer, e.g. make sql Q=\"SELECT count(*) FROM clean.coffee_reviews\""

install:
	uv sync
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

extract:
	uv run coffee-mlops extract --domain $(DOMAIN)

validate:
	uv run coffee-mlops validate --domain $(DOMAIN)

clean-layer:
	uv run coffee-mlops clean --domain $(DOMAIN)

features:
	uv run coffee-mlops features --domain $(DOMAIN)

sql:
	uv run coffee-mlops sql "$(Q)" --domain $(DOMAIN)
