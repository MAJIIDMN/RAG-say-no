from ragkit.core.registry import GENERATORS
from ragkit.core.schemas import Chunk, Claim, ClaimCitation, GeneratedAnswer, ScoredChunk
from ragkit.generation import prompts
from ragkit.llm.base import StructuredOutputError
from ragkit.llm.fake import FakeLLMClient

CONTEXT = [
    ScoredChunk(
        chunk=Chunk(
            chunk_id="bg::c0000",
            doc_id="bg",
            title="Background Tasks",
            section="Background Tasks > Using BackgroundTasks",
            text="Import BackgroundTasks. Add tasks with add_task. They run after the response is sent.",
        ),
        score=0.9,
    ),
    ScoredChunk(
        chunk=Chunk(chunk_id="cors::c0000", doc_id="cors", text="CORS settings."), score=0.2
    ),
]


def test_context_labels_passages_in_rank_order():
    user = prompts.answer_user("q", CONTEXT)
    assert user.index("[S1]") < user.index("[S2]")
    assert "Background Tasks > Using BackgroundTasks" in user


def test_extractive_answers_with_the_best_matching_sentence_as_quote():
    gen = GENERATORS.create({"impl": "extractive"}).generate(
        "Do tasks run after the response is sent?", CONTEXT
    )
    claim = gen.answer.claims[0]
    assert gen.answer.status == "answered"
    assert claim.text == "They run after the response is sent."
    assert claim.citations[0].source_id == "S1" and claim.citations[0].quote == claim.text


def test_extractive_reports_insufficient_context_without_overlap():
    gen = GENERATORS.create({"impl": "extractive"}).generate("kubernetes helm chart", CONTEXT)
    assert gen.answer.status == "insufficient_context" and gen.answer.claims == []


def test_llm_generator_uses_grounded_prompt_and_returns_usage():
    answer = GeneratedAnswer(
        status="answered",
        claims=[Claim(text="x", citations=[ClaimCitation(source_id="S1", quote="add_task")])],
    )
    client = FakeLLMClient(lambda s, u, schema: answer)
    gen = GENERATORS.create({"impl": "llm"}, llm=client).generate("q", CONTEXT)
    assert gen.answer == answer and gen.error is None
    assert "insufficient_context" in client.calls[0][0]


def test_llm_generator_naive_prompt_never_mentions_abstaining():
    client = FakeLLMClient()
    GENERATORS.create({"impl": "llm", "allow_abstain": False}, llm=client).generate("q", CONTEXT)
    assert "insufficient_context" not in client.calls[0][0]


def test_llm_generator_turns_model_failure_into_error_not_exception():
    def broken(s, u, schema):
        raise StructuredOutputError("json_object: invalid output")

    gen = GENERATORS.create({"impl": "llm"}, llm=FakeLLMClient(broken)).generate("q", CONTEXT)
    assert gen.answer is None and "StructuredOutputError" in gen.error
