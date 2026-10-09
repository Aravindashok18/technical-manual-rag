"""FastAPI service.

    uvicorn rag.service.api:app --port 8000

POST /ask       {"question": "...", "k": 4}  -> answer, cited sources, per-stage timings
GET  /health    index size, models, whether Ollama is reachable
GET  /metrics   rolling latency percentiles (JSON); /metrics/prometheus for text format
"""
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..embeddings import Embedder
from ..index import VectorIndex
from ..llm import OllamaClient
from ..pipeline import RagPipeline
from ..vision import DiagramReader
from .monitor import LatencyMonitor

log = logging.getLogger("rag.api")
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", Path(__file__).resolve().parents[2] / "data" / "index" / "default"))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


def load_pipeline():
    from ..llm import VISION_MODEL
    index = VectorIndex.load(INDEX_DIR)
    embedder = Embedder(index.embedder_key)
    threshold = float(os.environ["RAG_THRESHOLD"]) if os.environ.get("RAG_THRESHOLD") else None
    vision = DiagramReader(OllamaClient(VISION_MODEL, num_ctx=4096)) if os.environ.get("RAG_VISION", "1") == "1" else None
    return RagPipeline(index, embedder, OllamaClient(), prompt=os.environ.get("RAG_PROMPT", "concise_structured"),
                       threshold=threshold, vision=vision)


def create_app(pipeline=None):
    app = FastAPI(title="Technical manual assistant")
    monitor = LatencyMonitor()
    state = {"pipeline": pipeline}

    def get_pipeline():
        if state["pipeline"] is None:
            state["pipeline"] = load_pipeline()
        return state["pipeline"]

    @app.middleware("http")
    async def timing(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-Id"] = rid
        response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
        log.info("request_id=%s method=%s path=%s status=%s ms=%.1f", rid, request.method,
                 request.url.path, response.status_code, ms)
        return response

    @app.get("/health")
    def health():
        p = get_pipeline()
        return {"status": "ok", "chunks": len(p.index.chunks), "embedder": p.index.embedder_key,
                "chunking": p.index.strategy, "llm_model": getattr(p.llm, "model", "fake"),
                "llm_reachable": p.llm.available() if hasattr(p.llm, "available") else True}

    @app.post("/ask")
    def ask(req: AskRequest):
        t0 = time.perf_counter()
        try:
            ans = get_pipeline().ask(req.question, req.k)
        except Exception as exc:        # LLM down, index missing, ...
            monitor.record((time.perf_counter() - t0) * 1000, error=True)
            log.exception("ask failed")
            raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}")
        monitor.record(ans.timings["total_ms"], ans.timings, declined=ans.declined)
        return {"question": ans.question, "answer": ans.answer, "declined": ans.declined,
                "reason": ans.reason, "sources": ans.sources,
                "cited_pages": [{"manual": m, "page": p} for m, p in ans.cited],
                "used_vision": ans.used_vision,
                "timings_ms": {k: round(v, 1) for k, v in ans.timings.items() if v is not None}}

    @app.get("/metrics")
    def metrics():
        return monitor.snapshot()

    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    def prom():
        return monitor.prometheus()

    return app


app = create_app()
