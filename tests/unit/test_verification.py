from ragkit.core.registry import VERIFIERS
from ragkit.core.schemas import Chunk, Claim, ClaimCitation, ScoredChunk
from ragkit.llm.fake import FakeLLMClient
from ragkit.verification.verifiers import Entailment, quote_in, resolve_citations

CTX = [
    ScoredChunk(
        chunk=Chunk(
            chunk_id="bg::c0",
            doc_id="bg",
            title="Background",
            text="Tasks run after\nthe response is sent.",
            metadata={"url": "https://x/bg/"},
        ),
        score=0.9,
    )
]


def claim(quote, source="S1"):
    return Claim(text="Tasks run later.", citations=[ClaimCitation(source_id=source, quote=quote)])


def test_quote_in_normalises_whitespace_case_and_edge_punctuation():
    assert quote_in(
        '"Tasks run after the RESPONSE is sent."', "x Tasks run after\n the response is sent. y"
    )
    assert not quote_in("before the response", "tasks run after the response")
    assert not quote_in("  ", "anything")


def test_quote_verifier_accepts_verbatim_and_rejects_invented_quotes():
    v = VERIFIERS.create({"impl": "quote"})
    assert v.verify([claim("after the response is sent"), claim("before the request")], CTX) == [
        True,
        False,
    ]


def test_unknown_source_label_is_unsupported():
    assert VERIFIERS.create({"impl": "quote"}).verify(
        [claim("Tasks run after", source="S9")], CTX
    ) == [False]


def test_llm_verifier_asks_once_per_claim_with_cited_text():
    llm = FakeLLMClient(lambda s, u, schema: Entailment(supported="response" in u, reason="r"))
    out = VERIFIERS.create({"impl": "llm"}, llm=llm).verify(
        [claim("x"), claim("y", source="S2")], CTX
    )
    assert out == [True, False] and len(llm.calls) == 1


def test_resolve_citations_maps_labels_to_chunks_and_flags_quotes():
    cites = resolve_citations([claim("after the response"), claim("made up")], CTX)
    assert [(c.chunk_id, c.verified, c.url) for c in cites] == [
        ("bg::c0", True, "https://x/bg/"),
        ("bg::c0", False, "https://x/bg/"),
    ]
