from fastapi.testclient import TestClient

from rag.index import VectorIndex
from rag.llm import FakeLLM
from rag.pipeline import RagPipeline
from rag.service.api import create_app
from rag.service.monitor import LatencyMonitor


def test_search_ranks_the_matching_chunk_first(index, embedder):
    hits = index.search(embedder.encode_query("vacuum breaker contact movement mm"), 3)
    assert hits[0][0].page == 53
    assert hits[0][1] >= hits[1][1] >= hits[2][1]


def test_index_save_and_load_round_trip(index, embedder, tmp_path):
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    q = embedder.encode_query("compressor grease cups")
    assert [(c.manual, c.page) for c, _ in loaded.search(q, 2)] == [(c.manual, c.page) for c, _ in index.search(q, 2)]
    assert loaded.embedder_key == "hash" and len(loaded.chunks) == 4


def make_client(index, embedder, reply="Weekly. Source: [FIST 4-1A, p.130]"):
    pipe = RagPipeline(index, embedder, FakeLLM(reply), k=2)
    return TestClient(create_app(pipe))


def test_health(index, embedder):
    r = make_client(index, embedder).get("/health")
    assert r.status_code == 200 and r.json()["chunks"] == 4


def test_ask_returns_answer_sources_and_timings(index, embedder):
    c = make_client(index, embedder)
    r = c.post("/ask", json={"question": "How often should compressor grease cups be checked?"})
    body = r.json()
    assert r.status_code == 200
    assert body["sources"][0]["page"] == 130 and body["cited_pages"] == [{"manual": "FIST 4-1A", "page": 130}]
    assert body["timings_ms"]["total_ms"] >= 0
    assert "X-Process-Time-ms" in r.headers and "X-Request-Id" in r.headers


def test_validation_rejects_short_questions(index, embedder):
    assert make_client(index, embedder).post("/ask", json={"question": "hi"}).status_code == 422


def test_metrics_count_requests_and_report_percentiles(index, embedder):
    c = make_client(index, embedder)
    for _ in range(5):
        c.post("/ask", json={"question": "How often should compressor grease cups be checked?"})
    m = c.get("/metrics").json()
    assert m["requests"] == 5 and m["errors"] == 0
    assert m["latency"]["count"] == 5 and m["latency"]["p95_ms"] >= m["latency"]["p50_ms"]
    assert "retrieve_ms" in m["stages"]
    assert "rag_requests_total 5" in c.get("/metrics/prometheus").text


def test_llm_failure_is_a_503_and_counted(index, embedder):
    class Down:
        model = "x"
        def chat(self, *a, **k): raise ConnectionError("ollama is down")
    c = TestClient(create_app(RagPipeline(index, embedder, Down(), k=2)))
    assert c.post("/ask", json={"question": "How often are grease cups checked?"}).status_code == 503
    assert c.get("/metrics").json()["errors"] == 1


def test_latency_monitor_percentiles():
    m = LatencyMonitor()
    for ms in range(1, 101):
        m.record(float(ms), {"retrieve_ms": 1.0})
    s = m.snapshot()["latency"]
    assert abs(s["p50_ms"] - 50.5) < 1 and abs(s["p95_ms"] - 95.05) < 1
