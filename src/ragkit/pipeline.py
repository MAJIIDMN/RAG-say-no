"""retrieve → rerank → gate → generate → gate → verify. Never raises for "no answer"."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from ragkit.abstention.policy import AbstentionPolicy, Decision
from ragkit.config import RAGConfig
from ragkit.core.registry import GENERATORS, RERANKERS, RETRIEVERS, VERIFIERS
from ragkit.core.schemas import RAGResponse, add_usage
from ragkit.core.stages import Embedder, LLMClient
from ragkit.index.store import Index, get_embedder, load_index
from ragkit.verification.verifiers import resolve_citations


class LazyLLM:
    """Builds the real client on first use, so configs that never call an LLM need no key."""

    def __init__(self, client: LLMClient | None = None, role: str = "generator"):
        self._client, self._role = client, role

    @property
    def model(self) -> str:
        return self._get().model

    def _get(self) -> LLMClient:
        if self._client is None:
            from ragkit.llm.base import client_for

            self._client = client_for(self._role)
        return self._client

    def generate_structured(self, system, user, schema):
        return self._get().generate_structured(system, user, schema)


class RAGPipeline:
    def __init__(
        self,
        cfg: RAGConfig,
        index: Index | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ):
        self.cfg = cfg
        self.index = index or load_index(cfg)
        self.llm = LazyLLM(llm)
        deps: dict[str, Any] = {
            "index": self.index,
            "embedder": embedder or get_embedder(cfg.embedder),
            "llm": self.llm,
        }
        self.retriever = RETRIEVERS.create(cfg.retriever, **deps)
        self.reranker = RERANKERS.create(cfg.reranker)
        self.generator = GENERATORS.create(cfg.generator, **deps)
        self.verifier = VERIFIERS.create(cfg.verifier, **deps)
        self.policy = AbstentionPolicy(cfg.abstention, self.llm)

    def answer(self, query: str) -> RAGResponse:
        cfg, timings, usage, diag = self.cfg, {}, {}, {}

        @contextmanager
        def timed(name: str):
            t = time.perf_counter()
            yield
            timings[name] = round((time.perf_counter() - t) * 1000, 1)

        t_total = time.perf_counter()
        with timed("retrieve"):
            candidates = self.retriever.search(query, cfg.retrieval.candidates)
        diag["candidate_ids"] = [c.chunk.chunk_id for c in candidates]
        with timed("rerank"):
            context = self.reranker.rerank(
                query, candidates[: cfg.retrieval.rerank_candidates], cfg.retrieval.context_size
            )
        confidence = context[0].score if context else 0.0
        diag["top_score"] = confidence

        def finish(decision: Decision | None = None, **kw) -> RAGResponse:
            timings["total"] = round((time.perf_counter() - t_total) * 1000, 1)
            if decision is not None:
                add_usage(usage, decision.usage)
                detail = f" ({decision.detail})" if decision.detail else ""
                kw = {
                    "answered": False,
                    "answer": cfg.abstention.message + detail,
                    "abstain_reason": decision.reason,
                }
            return RAGResponse(
                query=query,
                confidence=confidence,
                retrieved=context,
                timings_ms=timings,
                usage=usage,
                diagnostics=diag,
                **kw,
            )

        with timed("gate_before"):
            decision = self.policy.before_generation(query, context)
        if decision.abstain:
            return finish(decision)
        add_usage(usage, decision.usage)

        with timed("generate"):
            generation = self.generator.generate(query, context)
        add_usage(usage, generation.usage)
        diag["status"] = generation.answer.status if generation.answer else None
        diag["generation_error"] = generation.error
        decision = self.policy.after_generation(generation)
        if decision.abstain:
            return finish(decision)

        claims = generation.answer.claims
        with timed("verify"):
            verification = self.verifier.verify(claims, context)
        add_usage(usage, verification.usage)
        if verification.error:
            return finish(Decision("verification_failed", verification.error))
        kept = [c for c, ok in zip(claims, verification.verdicts, strict=True) if ok]
        diag["claims_total"], diag["claims_supported"] = len(claims), len(kept)
        if not kept:
            return finish(Decision("unsupported_claims"))
        return finish(
            answered=True,
            answer=" ".join(c.text.strip() for c in kept),
            claims=kept,
            citations=resolve_citations(kept, context),
        )
