"""
mlops.py
---------
Endpoints for the Dashboard and MLOps Learning Loop.
"""

import sys
from pathlib import Path
import json
import csv
import subprocess
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from api.database import get_supabase

router = APIRouter()
ROOT = Path(__file__).resolve().parents[2]

LOG_FILE = ROOT / "logs" / "app.log"
FEEDBACK_FILE = ROOT / "data" / "raw" / "feedback.csv"

class FeedbackRequest(BaseModel):
    query: str
    expected_intent: str
    expected_priority: str

@router.get("/analytics", tags=["MLOps"])
def get_analytics():
    """Parse app.log to get dashboard metrics."""
    counts = {"complaints": 0, "high_priority": 0, "total": 0}
    
    db = get_supabase()
    if db:
        try:
            res = db.table("classified_queries").select("intent, priority").execute()
            data = res.data
            counts["total"] = len(data)
            counts["complaints"] = sum(1 for row in data if row.get("intent") == "complaint")
            counts["high_priority"] = sum(1 for row in data if row.get("priority") == "high")
            return counts
        except Exception:
            pass
            
    if not LOG_FILE.exists():
        return counts

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "Prediction complete" in line:
                try:
                    data = json.loads(line)
                    counts["total"] += 1
                    if data.get("intent") == "complaint":
                        counts["complaints"] += 1
                    if data.get("priority") == "high":
                        counts["high_priority"] += 1
                except Exception:
                    pass
    return counts

@router.post("/feedback", tags=["MLOps"])
def submit_feedback(req: FeedbackRequest):
    """Store incorrect predictions in feedback.csv and Supabase."""
    
    db = get_supabase()
    if db:
        try:
            db.table("feedback").insert({
                "query": req.query,
                "expected_intent": req.expected_intent,
                "expected_priority": req.expected_priority,
                "source": "user_feedback"
            }).execute()
        except Exception:
            pass

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = FEEDBACK_FILE.exists()
    
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["query", "intent", "priority", "source"])
        writer.writerow([req.query, req.expected_intent, req.expected_priority, "user_feedback"])
        
    return {"status": "success", "message": "Feedback recorded. The system will learn from this."}

@router.post("/retrain", tags=["MLOps"])
def trigger_retraining(background_tasks: BackgroundTasks):
    """Trigger the training script in the background."""
    def run_training():
        script_path = ROOT / "training" / "scripts" / "train.py"
        subprocess.Popen([sys.executable, str(script_path)])
        
    background_tasks.add_task(run_training)
    return {"status": "success", "message": "Retraining started. The system is improving."}
