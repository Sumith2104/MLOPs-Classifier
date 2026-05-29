"""
test_api.py
------------
Integration tests for the FastAPI /predict and /health endpoints.

Run:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_supabase_db(monkeypatch):
    """Bypass Supabase calls during API integration tests."""
    monkeypatch.setattr("api.routes.predict.get_supabase", lambda: None)
    monkeypatch.setattr("api.routes.mlops.get_supabase", lambda: None)


VALID_INTENTS    = {"complaint", "inquiry", "feedback"}
VALID_PRIORITIES = {"high", "medium", "low"}


# ── Health ────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_body(self):
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"


# ── Root ──────────────────────────────────────────────────────────────────────
class TestRoot:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_endpoints_key(self):
        resp = client.get("/")
        assert "endpoints" in resp.json()


# ── Predict ───────────────────────────────────────────────────────────────────
class TestPredict:
    def test_valid_request_200(self):
        resp = client.post("/predict", json={"query": "My product arrived damaged."})
        assert resp.status_code == 200

    def test_response_has_all_fields(self):
        resp = client.post("/predict", json={"query": "What is your refund policy?"})
        body = resp.json()
        for field in ["query", "intent", "priority", "intent_confidence",
                      "priority_confidence", "flagged", "timestamp"]:
            assert field in body, f"Missing field: {field}"

    def test_intent_is_valid(self):
        resp = client.post("/predict", json={"query": "I love the fast delivery!"})
        assert resp.json()["intent"] in VALID_INTENTS

    def test_priority_is_valid(self):
        resp = client.post("/predict", json={"query": "URGENT: account hacked!"})
        assert resp.json()["priority"] in VALID_PRIORITIES

    def test_flagged_is_boolean(self):
        resp = client.post("/predict", json={"query": "Need help with my order."})
        assert isinstance(resp.json()["flagged"], bool)

    def test_confidence_in_range(self):
        resp = client.post("/predict", json={"query": "Can I change my shipping address?"})
        body = resp.json()
        assert 0.0 <= body["intent_confidence"] <= 1.0
        assert 0.0 <= body["priority_confidence"] <= 1.0


# ── Validation errors ─────────────────────────────────────────────────────────
class TestPredictValidation:
    def test_missing_query_field(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_empty_string_query(self):
        resp = client.post("/predict", json={"query": "  "})
        # Server-side strip → 422
        assert resp.status_code in (422, 200)

    def test_query_too_short(self):
        resp = client.post("/predict", json={"query": "Hi"})
        assert resp.status_code == 422


# ── Batch Predict ─────────────────────────────────────────────────────────────
class TestBatchPredict:
    def test_batch_valid_request_200(self):
        resp = client.post(
            "/predict/batch",
            json={"queries": ["My product is damaged.", "Where is my refund?"]}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert body["total"] == 2
        assert isinstance(body["results"], list)
        assert len(body["results"]) == 2
        for r in body["results"]:
            assert "intent" in r
            assert "priority" in r
            assert "query" in r

    def test_batch_empty_queries(self):
        resp = client.post("/predict/batch", json={"queries": []})
        assert resp.status_code == 422

    def test_batch_duplicate_queries(self):
        resp = client.post(
            "/predict/batch",
            json={"queries": ["My product is damaged.", "My product is damaged."]}
        )
        assert resp.status_code == 422

    def test_batch_too_many_queries(self):
        resp = client.post(
            "/predict/batch",
            json={"queries": [f"Query text {i}" for i in range(51)]}
        )
        assert resp.status_code == 422

    def test_batch_query_too_short(self):
        resp = client.post(
            "/predict/batch",
            json={"queries": ["Hi", "Where is my refund?"]}
        )
        assert resp.status_code == 422

    def test_batch_concurrency_limit_429(self, monkeypatch):
        from api.routes.predict import get_batch_semaphore
        sem = get_batch_semaphore()
        monkeypatch.setattr(sem, "locked", lambda: True)
        
        resp = client.post(
            "/predict/batch",
            json={"queries": ["My product is damaged.", "Where is my refund?"]}
        )
        assert resp.status_code == 429
        assert "Too many concurrent batch requests" in resp.json()["detail"]

    def test_batch_timeout_504(self, monkeypatch):
        from model.predictor import Predictor
        import time
        def mock_predict_batch_slow(self, texts):
            time.sleep(1.0)
            return [{"intent": "complaint", "priority": "high", "intent_confidence": 0.9, "priority_confidence": 0.9, "flagged": False} for _ in texts]
            
        monkeypatch.setattr(Predictor, "predict_batch", mock_predict_batch_slow)
        
        from api.routes.predict import get_timeout_seconds
        monkeypatch.setattr("api.routes.predict.get_timeout_seconds", lambda: 0.1)
        
        resp = client.post(
            "/predict/batch",
            json={"queries": ["My product is damaged.", "Where is my refund?"]}
        )
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]

    def test_rate_limiter_blocks_requests(self, monkeypatch):
        from api.middleware.rate_limiter import InMemoryRateLimiter
        original_is_allowed = InMemoryRateLimiter.is_allowed
        monkeypatch.setattr(InMemoryRateLimiter, "is_allowed", lambda self, cid: (False, 30))
        
        try:
            resp = client.post("/predict", json={"query": "My product arrived damaged."})
            assert resp.status_code == 429
            assert "Too many requests" in resp.json()["detail"]
            assert resp.headers.get("Retry-After") == "30"
        finally:
            monkeypatch.setattr(InMemoryRateLimiter, "is_allowed", original_is_allowed)
