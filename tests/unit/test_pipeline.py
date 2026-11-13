import pytest

from ragkit.config import RAGConfig
from ragkit.core.schemas import Claim, ClaimCitation, GeneratedAnswer
from ragkit.llm.fake import FakeLLMClient
from ragkit.pipeline import RAGPipeline


def config(**over):
    base = {
        "retriever": {"impl": "hybrid"},
        "reranker": {"impl": "lexical"},
        "generator": {"impl": "extractive"},
        "verifier": {"impl": "quote"},
        "abstention": {"min_score": 0.3},
    }
    return RAGConfig.model_validate({**base, **over})


@pytest.fixture
def pipe(tiny_index):
    index, emb = tiny_index

    def make(cfg=None, llm=None):
        return RAGPipeline(cfg or config(), index=index, embedder=emb, llm=llm)

    return make


def test_answers_with_verified_citation_offline(pipe):
    resp = pipe().answer("When do BackgroundTasks run after the response?")
    assert resp.answered and resp.abstain_reason is None
    assert resp.citations and all(c.verified for c in resp.citations)
    assert resp.citations[0].doc_id == "bg"
    assert {"retrieve", "rerank", "generate", "verify", "total"} <= set(resp.timings_ms)
    assert resp.diagnostics["candidate_ids"]


def test_off_topic_question_abstains_at_the_score_gate(pipe):
    resp = pipe().answer("kubernetes helm chart rollout")
    assert not resp.answered and resp.abstain_reason in {
        "low_retrieval_score",
        "no_retrieval_results",
    }
    assert resp.answer.startswith("The documentation does not contain")


def test_fabricated_quote_is_dropped_and_turns_into_abstention(pipe):
    fabricated = GeneratedAnswer(
        status="answered",
        claims=[
            Claim(
                text="It uses Celery.",
                citations=[ClaimCitation(source_id="S1", quote="uses Celery")],
            )
        ],
    )
    cfg = config(generator={"impl": "llm"})
    resp = pipe(cfg, FakeLLMClient(lambda s, u, schema: fabricated)).answer(
        "BackgroundTasks response"
    )
    assert resp.abstain_reason == "unsupported_claims"
    assert resp.diagnostics["claims_total"] == 1 and resp.diagnostics["claims_supported"] == 0


def test_llm_is_never_built_for_the_offline_config(pipe, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "")
    assert pipe().answer("CORSMiddleware origins").answered
