"""
logger.py
----------
Structured JSON logger for all API requests and predictions.
Output goes to both stdout and logs/app.log.

Usage:
    from monitoring.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Prediction made", extra={"intent": "complaint"})
"""

import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR  = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


class JsonFormatter(logging.Formatter):
    """Format every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        # Merge any extra fields passed via the `extra` kwarg
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in payload:
                payload[key] = val

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    formatter = JsonFormatter()

    # Stream handler (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
