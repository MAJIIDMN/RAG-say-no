from ragkit.core.registry import JUDGES
from ragkit.eval.judges import Correctness
from ragkit.llm.fake import FakeLLMClient


def test_lexical_judge_checks_reference_content_words():
    j = JUDGES.create({"impl": "lexical"})
    assert j.judge("q", "Up to 3 times by default.", "Jobs are retried 3 times by default.").correct
    assert not j.judge("q", "Up to 3 times by default.", "Jobs never retry.").correct


def test_lexical_judge_needs_a_number_match_when_the_reference_has_one():
    j = JUDGES.create({"impl": "lexical", "min_recall": 0.1})
    assert not j.judge("q", "180 seconds.", "The timeout is 60 seconds.").correct


def test_llm_judge_returns_verdict_reason_and_usage():
    llm = FakeLLMClient(lambda s, u, schema: Correctness(correct="42" in u, reason="matches"))
    out = JUDGES.create({"impl": "llm"}, llm=llm).judge("q", "42", "It is 42.")
    assert out.correct and out.reason == "matches" and len(llm.calls) == 1
