# Keep system-wide PYTHONPATH entries (e.g. ROS) out of the project venv.
unexport PYTHONPATH
# --no-sync: keep whichever extras `make setup` / `make setup-ml` installed.
RUN := uv run --no-sync
CONFIG ?= configs/toy.yaml
EVALSET ?= evals/datasets/toy.jsonl
SPLIT ?= test
ABLATION_SET ?= evals/datasets/fastapi.jsonl

.PHONY: setup setup-ml hooks test test-slow lint format data ingest eval tune ablation report serve

setup:            ## core + LLM SDKs + plotting + API + dev tools; enough for tests and the offline baseline
	uv sync --extra llm --extra evals --extra api
	@test -f .env || cp .env.example .env

setup-ml:         ## + sentence-transformers and CPU torch for real embedders/rerankers
	uv sync --extra llm --extra ml --extra evals --extra api
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

ingest:           ## build the index for CONFIG (default: offline toy)
	$(RUN) ragkit ingest -c $(CONFIG)

eval: ingest      ## run EVALSET/SPLIT through CONFIG and write runs/<stamp>_<name>_<split>/
	$(RUN) ragkit eval -c $(CONFIG) -e $(EVALSET) --split $(SPLIT)

tune: ingest      ## pick the score threshold on the calib split, write configs/tuned/
	$(RUN) ragkit tune -c $(CONFIG) -e $(EVALSET)

ablation:         ## E001-E007 on ABLATION_SET/SPLIT (needs make data, setup-ml and LLM_* in .env)
	@for c in configs/experiments/E*.yaml; do \
		$(RUN) ragkit ingest -c $$c && $(RUN) ragkit eval -c $$c -e $(ABLATION_SET) --split $(SPLIT) || exit 1; \
	done

report:           ## ablation table from the newest run per config
	$(RUN) ragkit report --split $(SPLIT) --prefix E --out docs/ablation_$(SPLIT).md

serve: ingest     ## POST /ask and GET /health on http://127.0.0.1:8000 for CONFIG
	$(RUN) ragkit serve -c $(CONFIG)
