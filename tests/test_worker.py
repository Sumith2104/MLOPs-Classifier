# tests/test_worker.py
import sys
import asyncio
from pathlib import Path
import pytest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.worker import start_query_classifier_worker
from model.predictor import Predictor

class MockSupabaseExecute:
    def __init__(self, data):
        self.data = data

@pytest.mark.anyio
async def test_worker_processing_flow(monkeypatch):
    # Mock data returned by select().eq().limit().execute()
    # Using 'message' column to match existing database schema
    mock_queries = [
        {"id": "uuid-1", "message": "My order arrived broken."},
        {"id": "uuid-2", "message": "What are your store hours?"}
    ]
    
    # Mock database queries and inserts
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_eq = MagicMock()
    mock_limit = MagicMock()
    
    mock_db.table.return_value = mock_table
    mock_table.select.return_value = mock_select
    mock_select.eq.return_value = mock_eq
    mock_eq.limit.return_value = mock_limit
    mock_limit.execute.return_value = MockSupabaseExecute(mock_queries)
    
    # Mock update chain
    mock_update = MagicMock()
    mock_in = MagicMock()
    mock_table.update.return_value = mock_update
    mock_update.in_.return_value = mock_in
    mock_in.execute.return_value = MockSupabaseExecute([])
    
    # Mock insert chain
    mock_insert = MagicMock()
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.return_value = MockSupabaseExecute([])
    
    # Patch database connection function
    monkeypatch.setattr("api.worker.get_supabase", lambda: mock_db)
    
    # Mock model predictions
    mock_pred_results = [
        {"intent": "complaint", "priority": "high", "intent_confidence": 0.95, "priority_confidence": 0.90, "flagged": False},
        {"intent": "inquiry", "priority": "low", "intent_confidence": 0.98, "priority_confidence": 0.95, "flagged": False}
    ]
    
    monkeypatch.setattr(Predictor, "predict_batch", lambda self, queries: mock_pred_results)
    
    # Ensure the worker exits after one loop run by raising CancelledError in sleep
    sleep_calls = 0
    async def mock_sleep(seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 1:
            raise asyncio.CancelledError("Simulated loop exit")
            
    monkeypatch.setattr(asyncio, "sleep", mock_sleep)
    
    # Execute the background task
    try:
        await start_query_classifier_worker()
    except asyncio.CancelledError:
        pass
        
    # Verify the database mock interactions
    mock_db.table.assert_any_call("client_queries")
    mock_table.select.assert_any_call("id, message")
    mock_select.eq.assert_called_with("processed", False)
    
    # Verify rows were marked as processed
    mock_table.update.assert_called_with({"processed": True})
    mock_update.in_.assert_called_with("id", ["uuid-1", "uuid-2"])
    
    # Verify predictions were written to classified_queries
    mock_db.table.assert_any_call("classified_queries")
    inserted_args = mock_table.insert.call_args[0][0]
    assert len(inserted_args) == 2
    assert inserted_args[0]["query_id"] == "uuid-1"
    assert inserted_args[0]["message"] == "My order arrived broken."
    assert inserted_args[0]["intent"] == "complaint"
    assert inserted_args[0]["priority"] == "high"
    assert inserted_args[1]["query_id"] == "uuid-2"
    assert inserted_args[1]["message"] == "What are your store hours?"
    assert inserted_args[1]["intent"] == "inquiry"
    assert inserted_args[1]["priority"] == "low"
