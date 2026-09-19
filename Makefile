DOMAIN ?= coffee

.PHONY: help install lint format typecheck test check

help:
	@echo "install    venv + dependencies + git hooks"
	@echo "lint       ruff lint and format check"
	@echo "format     apply ruff fixes and formatting"
	@echo "typecheck  mypy --strict"
	@echo "test       pytest (offline tests only)"
	@echo "check      lint + typecheck + test (what CI runs)"

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

check: lint typecheck test
