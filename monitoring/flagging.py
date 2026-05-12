"""
flagging.py
------------
Confidence-based flagging logic for human review.

A prediction is flagged when either the intent or priority confidence
score falls below the configured threshold (default: 0.70).

Flagged predictions are appended to logs/flagged_predictions.jsonl for
manual review and potential retraining.

Usage:
    from monitoring.flagging import FlagManager
    fm = FlagManager(threshold=0.70)
    flagged = fm.check_and_log(query, prediction_result)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from monitoring.logger import get_logger

LOG_DIR      = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)
FLAGGED_FILE = LOG_DIR / "flagged_predictions.jsonl"

logger = get_logger(__name__)


class FlagManager:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def is_flagged(self, intent_conf: float, priority_conf: float) -> bool:
        return (intent_conf < self.threshold) or (priority_conf < self.threshold)

    def log_flagged(self, query: str, result: dict) -> None:
        """Append a flagged prediction to the JSONL log file."""
        record = {
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "query":               query,
            "intent":              result.get("intent"),
            "priority":            result.get("priority"),
            "intent_confidence":   result.get("intent_confidence"),
            "priority_confidence": result.get("priority_confidence"),
        }
        with open(FLAGGED_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.warning(
            "Low-confidence prediction flagged for human review",
            extra={
                "intent_conf":   result.get("intent_confidence"),
                "priority_conf": result.get("priority_confidence"),
                "threshold":     self.threshold,
            },
        )

    def check_and_log(self, query: str, result: dict) -> bool:
        """
        Check confidence; if flagged, persist to JSONL log.
        Returns True if flagged.
        """
        flagged = self.is_flagged(
            result.get("intent_confidence",   1.0),
            result.get("priority_confidence", 1.0),
        )
        if flagged:
            self.log_flagged(query, result)
        return flagged
