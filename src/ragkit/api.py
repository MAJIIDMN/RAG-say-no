"""HTTP API (needs the `api` extra). `ragkit serve -c <config>` runs it with uvicorn."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from ragkit.pipeline import RAGPipeline


class AskRequest(BaseModel):
    question: str = Field(max_length=2000)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


def create_app(pipeline: RAGPipeline) -> FastAPI:
    app = FastAPI(title="ragkit", description="Grounded answers with citations, or an abstention.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "config": pipeline.cfg.name, "chunks": pipeline.index.size}

    @app.post("/ask")
    async def ask(req: AskRequest, include_passages: bool = False) -> dict[str, Any]:
        resp = await run_in_threadpool(pipeline.answer, req.question)
        return resp.model_dump(exclude=None if include_passages else {"retrieved"})

    return app
