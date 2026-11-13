# Keep system-wide PYTHONPATH entries (e.g. ROS) out of the project venv.
unexport PYTHONPATH
# --no-sync: keep whichever extras `make setup` / `make setup-ml` installed.
RUN := uv run --no-sync

.PHONY: setup setup-ml hooks test test-slow lint format

setup:            ## core + LLM SDKs + dev tools; enough for tests and the offline baseline
	uv sync --extra llm
	@test -f .env || cp .env.example .env

setup-ml:         ## + sentence-transformers and CPU torch for real embedders/rerankers
	uv sync --extra llm --extra ml
	@test -f .env || cp .env.example .env

hooks:
	$(RUN) pre-commit install

test:
	$(RUN) pytest

test-slow:        ## real embedding/reranker models (needs setup-ml, downloads ~200 MB once)
	RAGKIT_SLOW=1 $(RUN) pytest tests/integration

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format:
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests
