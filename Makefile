# Keep system-wide PYTHONPATH entries (e.g. ROS) out of the project venv.
unexport PYTHONPATH

.PHONY: setup hooks test lint format

setup:
	uv sync
	@test -f .env || cp .env.example .env

hooks:
	uvx pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests
