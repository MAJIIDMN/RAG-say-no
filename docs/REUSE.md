# Reusing `ragkit` for the next RAG project

This repo has two roles: the "Grounded RAG" project over the FastAPI docs, and the `ragkit`
foundation for other RAG projects. This guide covers what to keep, what to replace, and how to add
components without touching the pipeline code. Design background is in
[architecture.md](architecture.md).

## Generic vs project-specific

| Location | Contents | In a new project |
|---|---|---|
| `src/ragkit/` | Schemas, registries, pipeline, abstention gates, verification, eval, CLI, API | Keep as is |
| `tests/unit/`, `tests/integration/` | Tests for `ragkit` | Keep as is |
| `Makefile`, `Dockerfile`, `.github/workflows/ci.yml` | Commands and CI | Change `CONFIG`, `EVALSET`, `ABLATION_SET` |
| `data/toy/`, `evals/datasets/toy.jsonl`, `configs/toy.yaml` | Fictional corpus for CI | Keep, so CI stays offline |
| `configs/base.yaml`, `configs/experiments/` | FastAPI corpus and variants E001–E007 | Rewrite |
| `scripts/prepare_fastapi_docs.py` | FastAPI preprocessing | Replace with a script for the new source |

The rule: no domain logic in `src/ragkit/`. If a change helps every project (say, a PDF-aware
chunker), it belongs in `src/ragkit/`. If it helps one project, it lives in that project's package
and loads through `plugins`.

## Starting a new project

1. Copy this repo, then delete `configs/experiments/` and `scripts/prepare_fastapi_docs.py`.
2. Write a preprocessing script that emits Markdown, text, HTML or PDF files with `title` and `url`
   front matter into `data/processed/<name>/`. The loader reads that front matter for citations.
3. Edit `configs/base.yaml`: `corpus.path`, chunker and models.
4. Write a JSONL eval set with `dev`, `calib` and `test` splits grouped by page, and run
   `ragkit evalset check -c configs/base.yaml -e evals/datasets/<name>.jsonl` until it reports
   `0 problems`.
5. Add one config per experiment that lists only its change (`extends`), then run `make ablation`
   and `make report`.

## Adding a stage implementation

Each stage is a `Protocol` in `src/ragkit/core/stages.py`: `Chunker`, `Embedder`, `Retriever`,
`Reranker`, `LLMClient`, `Generator`, `Verifier` and `Judge`. An implementation registers under a
name with a params model, and a config picks it with `{impl: name, ...params}`. Unknown params are
rejected when the config loads.

A project-specific reranker in `src/my_project/stages.py`:

```python
from pydantic import BaseModel, ConfigDict

from ragkit.core.registry import RERANKERS
from ragkit.core.schemas import ScoredChunk


class TitleBoostParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    boost: float = 0.2


class TitleBoostReranker:
    """Keeps the retriever order but favours passages whose title shares a query word."""

    def __init__(self, params: TitleBoostParams):
        self.p = params

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        words = set(query.lower().split())
        scored = [
            c.model_copy(
                update={"score": c.score + self.p.boost * bool(words & set(c.chunk.title.lower().split()))}
            )
            for c in candidates
        ]
        return sorted(scored, key=lambda c: c.score, reverse=True)[:top_n]


@RERANKERS.register("title_boost", TitleBoostParams)
def _title_boost(params: TitleBoostParams, **_) -> TitleBoostReranker:
    return TitleBoostReranker(params)
```

Add the package to `pyproject.toml` (`[tool.hatch.build.targets.wheel] packages =
["src/ragkit", "src/my_project"]`) and load it from the config:

```yaml
extends: base.yaml
name: E008_title_boost
plugins: [my_project.stages]
reranker: {impl: title_boost, boost: 0.3}
```

Factories receive dependencies as keywords (`index`, `embedder`, `llm`); take what you need and
swallow the rest with `**_`. The example above was run on the toy corpus with `ragkit ingest` and
`ragkit ask`.

For a generic implementation inside `src/ragkit/`, import its module in `src/ragkit/builtins.py` so
it registers without `plugins`, and add tests under `tests/unit/`.

## Adding an LLM provider

`openai_compatible` covers endpoints that follow the OpenAI API, and `anthropic` uses the Anthropic
SDK. Another provider registers in `LLM_CLIENTS` with a factory that takes `settings: LLMSettings`
and returns an object whose `generate_structured(system, user, schema)` returns
`(pydantic_object, usage)`. Select it with `LLM_PROVIDER=<name>` in `.env`.

## Things to keep in mind

- Bump `PROMPT_VERSION` in `src/ragkit/generation/prompts.py` on every prompt change; every run
  records it.
- `ragkit tune` reads only the `calib` split. Never tune anything on `test`.
- The index is cached by a hash of corpus, chunker and embedder. Variants that change only the
  retriever, reranker or generator reuse it. Editing corpus files triggers a rebuild on the next
  `ragkit ingest`.
- Put a `source` field (for example `mydocs@<commit>`) in each file's front matter. Index manifests
  and run metadata record it, so every number traces back to a corpus version.
- Keep working notes (PRD, ADRs, experiment log) in the gitignored `artifacts/` folder and summarize
  decisions worth publishing in `docs/architecture.md`.
