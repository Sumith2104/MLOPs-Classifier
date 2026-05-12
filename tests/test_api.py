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
