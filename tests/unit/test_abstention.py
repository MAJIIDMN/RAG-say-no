from ragkit.abstention.policy import AbstentionPolicy, Sufficiency
from ragkit.config import AbstentionConfig
from ragkit.core.schemas import (
    Chunk,
    Claim,
    ClaimCitation,
    GeneratedAnswer,
    Generation,
    ScoredChunk,
)
from ragkit.llm.fake import FakeLLMClient

CTX = [ScoredChunk(chunk=Chunk(chunk_id="a", doc_id="a", text="t"), score=0.4)]
ANSWERED = Generation(
    answer=GeneratedAnswer(
        status="answered",
        claims=[Claim(text="x", citations=[ClaimCitation(source_id="S1", quote="t")])],
    )
)
INSUFFICIENT = Generation(
    answer=GeneratedAnswer(status="insufficient_context", missing_information="no port")
)


def test_low_score_abstains_before_any_llm_call():
    llm = FakeLLMClient()
    policy = AbstentionPolicy(AbstentionConfig(min_score=0.5, sufficiency_check=True), llm)
    decision = policy.before_generation("q", CTX)
    assert decision.reason == "low_retrieval_score" and llm.calls == []


def test_empty_context_always_abstains_even_when_disabled():
    assert AbstentionPolicy(AbstentionConfig(enabled=False)).before_generation("q", []).reason == (
        "no_retrieval_results"
    )


def test_sufficiency_check_can_reject():
    llm = FakeLLMClient(lambda s, u, schema: Sufficiency(sufficient=False, reason="no port number"))
    decision = AbstentionPolicy(AbstentionConfig(sufficiency_check=True), llm).before_generation(
        "q", CTX
    )
    assert decision.reason == "insufficient_context" and decision.detail == "no port number"


def test_generator_status_is_respected_only_when_enabled():
    on = AbstentionPolicy(AbstentionConfig()).after_generation(INSUFFICIENT)
    off = AbstentionPolicy(AbstentionConfig(enabled=False)).after_generation(INSUFFICIENT)
    assert on.reason == "llm_insufficient_context" and on.detail == "no port"
    assert off.reason == "empty_answer"
    assert AbstentionPolicy(AbstentionConfig()).after_generation(ANSWERED).reason is None


def test_model_failure_is_its_own_reason():
    failed = Generation(answer=None, error="StructuredOutputError: bad json")
    decision = AbstentionPolicy(AbstentionConfig(enabled=False)).after_generation(failed)
    assert decision.reason == "generation_failed" and "bad json" in decision.detail
