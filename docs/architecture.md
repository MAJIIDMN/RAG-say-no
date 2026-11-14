# Architecture

How `ragkit` answers a question, where each stage can be swapped, and the design decisions behind
the defaults. Every decision below had a real alternative; the alternative is named so it can be
revisited with data.

## Pipeline

```text
question
  → retrieve   BM25 / dense / hybrid RRF          top 50 candidates
  → rerank     none / lexical / cross-encoder      top 30 scored, top 5 kept
  → gate       no candidates / top score < τ ?     abstain: no_retrieval_results / low_retrieval_score
  → gate       optional LLM sufficiency check      abstain: insufficient_context
  → generate   extractive / LLM structured output  claims + passage IDs + quotes
  → gate       status == insufficient_context ?    abstain: llm_insufficient_context
  → verify     none / verbatim quote / LLM         drop unsupported claims
  → gate       no claim left ?                     abstain: unsupported_claims
  → answer + citations (title, URL, quote)
```

`RAGPipeline.answer()` never raises for "no answer". Every refusal carries an `abstain_reason`, and
the response keeps raw signals (`diagnostics`, top score, candidate IDs) so the eval can replay the
score gate offline.

## Seams

Each stage is a `typing.Protocol` in `src/ragkit/core/stages.py`: `Chunker`, `Embedder`,
`Retriever`, `Reranker`, `LLMClient`, `Generator`, `Verifier`, `Judge`. Implementations register in
a name → factory registry with a Pydantic params model; a config picks one with
`{impl: name, ...params}` and unknown params fail at load time. Every seam has an offline adapter
(`hashing`, `lexical`, `extractive`, `fake`), so tests and CI need no GPU, model download or API
key. Seams with a single adapter (vector store, document loader) have no `Protocol` yet.

Configs are layered YAML (`extends`, deep merge, child wins; a changed `impl` replaces the whole
stage). Secrets and model choice come from the environment (`LLM_*`, `JUDGE_*`).

## Decisions

**A small package of our own instead of LangChain or LlamaIndex.** Every stage stays transparent and
measurable on its own, which an ablation study needs. The cost is writing advanced features
(agentic retrieval, graphs) as new adapters.

**Brute-force numpy index instead of FAISS, Qdrant or LanceDB.** For 1k–50k chunks a flat index is
exact and fast enough; the
[FAISS guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
note that the flat index is the only exact one. The index is cached by a hash of corpus, chunker and
embedder, so variants that only change later stages share it. The manifest also stores a digest of
the corpus text, and `ragkit ingest` rebuilds the index when that digest changes. Move to FAISS or Qdrant past about
100k chunks or when metadata filters are needed.

**Heading-aware chunking (~400 tokens, no overlap) with a breadcrumb prefix.** It follows the
document structure, keeps code blocks whole, and prefixes `title > H2 > H3` to the indexed text as a
deterministic stand-in for
[contextual retrieval](https://www.anthropic.com/news/contextual-retrieval). Fixed windows remain
for comparison (E007). Token counts use a regex tokenizer, so 400 is approximate; on FastAPI the
mean chunk is 221.8 tokens and p95 is 394.

**bge-small-en-v1.5 + ms-marco-MiniLM-L6-v2 by default.** Both are small enough for CPU (33.4M and
22.7M parameters). bge-m3 with bge-reranker-v2-m3, and gte-modernbert, are candidate experiments; the
choice follows recall@k on `dev`, not a leaderboard. The cross-encoder score gets an explicit
sigmoid so the abstention threshold lives in [0, 1]; the threshold must be retuned whenever the
reranker changes.

**RRF with k = 60 instead of weighted score fusion.** RRF ignores score scales, and k = 60 comes
from [Cormack et al. (2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). It is
written explicitly in config because defaults differ across tools. The RRF score is never used as
confidence; that comes from the reranker.

**Layered abstention instead of a single threshold or a single LLM check.** The score gate is free
and runs before any LLM call; the generator's `insufficient_context` status (the
[Sufficient Context](https://arxiv.org/abs/2411.06037) idea) catches relevant but incomplete
context; verification catches unsupported claims. A parse failure counts as an abstention with
reason `generation_failed`.

**Verbatim quotes checked deterministically, with an optional LLM verifier.** A quote check is free
and testable but does not prove the quote supports the claim; the judge-based faithfulness metric
measures that gap during eval. NLI checkers such as MiniCheck wait until the LLM verifier is
measured.

**Our own `LLMClient.generate_structured()` instead of LiteLLM or instructor.** Structured output
support differs by provider, so the `openai_compatible` adapter tries `json_schema`, then
`json_object` with the schema in the prompt, then plain text with JSON extraction, and always
validates with Pydantic. The `anthropic` adapter uses the SDK's `messages.parse`. Schemas stay flat
to pass strict-mode limits.

**Our own metrics instead of Ragas or DeepEval as the source of truth.** The project's core numbers
(answers to unanswerable questions, over-refusal, the coverage curve, a CRAG-style 1/0/−1 score,
AURC) are not standard in those libraries. All metrics are pure functions with unit tests. LLM judge
numbers are reported only after an audit against 50 hand labels.

## Eval protocol

Eval sets are JSONL. Each item has `split` (`dev` for prompts and chunking, `calib` for the
threshold only, `test` for reporting only), `group` (the source page; a group never spans splits),
`answerable`, `category` and `evidence` (verbatim spans of the documents, so relevance survives a
chunking change). The target mix is 25–30% unanswerable, mostly near-misses: the topic is in the
docs, the asked fact is not. `ragkit evalset check` enforces the schema, unique IDs, no group
leakage and that every evidence span exists in the corpus; `ragkit tune` refuses anything but
`calib`.
