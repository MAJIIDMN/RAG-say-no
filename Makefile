# Keep system-wide PYTHONPATH entries (e.g. ROS) out of the project venv.
unexport PYTHONPATH
# --no-sync: keep whichever extras `make setup` / `make setup-ml` installed.
RUN := uv run --no-sync

.PHONY: setup setup-ml hooks test test-slow lint format data

setup:            ## core + LLM SDKs + plotting + dev tools; enough for tests and the offline baseline
	uv sync --extra llm --extra evals
	@test -f .env || cp .env.example .env

setup-ml:         ## + sentence-transformers and CPU torch for real embedders/rerankers
	uv sync --extra llm --extra ml --extra evals
	@test -f .env || cp .env.example .env

hooks:
	$(RUN) pre-commit install

test:
	$(RUN) pytest

test-slow:        ## real embedding/reranker models (needs setup-ml, downloads ~200 MB once)
	RAGKIT_SLOW=1 $(RUN) pytest tests/integration

lint:
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

format:
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts

data:             ## clone FastAPI docs at the pinned commit into data/raw and preprocess to data/processed
	$(RUN) python scripts/prepare_fastapi_docs.py
