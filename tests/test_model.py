"""
test_model.py
--------------
Unit tests for the Predictor inference pipeline.

Run:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.predictor import Predictor

VALID_INTENTS   = {"complaint", "inquiry", "feedback"}
VALID_PRIORITIES = {"high", "medium", "low"}

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def predictor():
    """Shared predictor instance across all tests in this module."""
    return Predictor()


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestPredictorOutput:
    def test_returns_dict(self, predictor):
        result = predictor.predict("My order is broken.")
        assert isinstance(result, dict)

    def test_all_keys_present(self, predictor):
        result = predictor.predict("What are your store hours?")
        expected_keys = {"intent", "priority", "intent_confidence", "priority_confidence", "flagged"}
        assert expected_keys.issubset(result.keys())

    def test_valid_intent_label(self, predictor):
        result = predictor.predict("I love the new feature, great job!")
        assert result["intent"] in VALID_INTENTS

    def test_valid_priority_label(self, predictor):
        result = predictor.predict("Urgent: my account is suspended!")
        assert result["priority"] in VALID_PRIORITIES

    def test_confidence_range(self, predictor):
        result = predictor.predict("Can you help me track my order?")
        assert 0.0 <= result["intent_confidence"] <= 1.0
        assert 0.0 <= result["priority_confidence"] <= 1.0

    def test_flagged_is_bool(self, predictor):
        result = predictor.predict("Some random query.")
        assert isinstance(result["flagged"], bool)


class TestPredictorBatch:
    def test_batch_returns_list(self, predictor):
        texts = ["Help!", "Where is my refund?", "Great service!"]
        results = predictor.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == len(texts)

    def test_batch_each_item_valid(self, predictor):
        texts = ["I was charged twice", "What is your return policy?"]
        results = predictor.predict_batch(texts)
        for r in results:
            assert r["intent"] in VALID_INTENTS
            assert r["priority"] in VALID_PRIORITIES


class TestEdgeCases:
    def test_short_input(self, predictor):
        result = predictor.predict("Hi!")
        assert result["intent"] in VALID_INTENTS

    def test_long_input(self, predictor):
        long_text = "My order is broken. " * 50
        result = predictor.predict(long_text)
        assert result["intent"] in VALID_INTENTS

    def test_special_chars(self, predictor):
        result = predictor.predict("Can't login!!! 😠 #urgent @support")
        assert result["intent"] in VALID_INTENTS
