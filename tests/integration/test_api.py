import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ragkit.api import create_app  # noqa: E402
from ragkit.config import RAGConfig  # noqa: E402
from ragkit.pipeline import RAGPipeline  # noqa: E402


@pytest.fixture
def client(tiny_index):
    index, emb = tiny_index
    cfg = RAGConfig.model_validate(
        {
            "name": "api-test",
            "reranker": {"impl": "lexical"},
            "verifier": {"impl": "quote"},
            "abstention": {"enabled": True, "min_score": 0.3},
        }
    )
    return TestClient(create_app(RAGPipeline(cfg, index=index, embedder=emb)))


def test_health_reports_config_and_index_size(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "config": "api-test", "chunks": 4}


def test_ask_answers_with_citations_and_hides_passages_by_default(client):
    body = client.post("/ask", json={"question": "When do BackgroundTasks run?"}).json()
    assert body["answered"] and body["citations"][0]["doc_id"] == "bg"
    assert "retrieved" not in body


def test_ask_abstains_off_topic_and_can_include_passages(client):
    r = client.post("/ask?include_passages=true", json={"question": "kubernetes helm rollout"})
    body = r.json()
    assert r.status_code == 200 and not body["answered"] and body["abstain_reason"]
    assert "retrieved" in body


def test_ask_rejects_an_empty_question(client):
    assert client.post("/ask", json={"question": "  "}).status_code == 422
