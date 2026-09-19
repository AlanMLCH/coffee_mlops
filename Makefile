DOMAIN ?= coffee

.PHONY: help install lint format typecheck test test-network check extract

help:
	@echo "install    venv + dependencies + git hooks"
	@echo "lint       ruff lint and format check"
	@echo "format     apply ruff fixes and formatting"
	@echo "typecheck  mypy --strict"
	@echo "test       pytest (offline tests only)"
	@echo "check      lint + typecheck + test (what CI runs)"
	@echo "test-network  check upstream URLs still answer (hits the internet)"
	@echo "extract    download DOMAIN sources to the raw layer (DOMAIN=$(DOMAIN))"

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
