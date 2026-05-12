"""
predict.py
-----------
FastAPI route for POST /predict and GET /health.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.schemas.query_schema import QueryRequest, PredictionResponse, HealthResponse
from model.predictor import Predictor
from monitoring.logger import get_logger
from monitoring.flagging import FlagManager

router = APIRouter()
logger = get_logger(__name__)

# Lazy-loaded singletons (initialised on first request)
_predictor: Predictor | None = None
_flag_manager: FlagManager | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def get_flag_manager() -> FlagManager:
    global _flag_manager
    if _flag_manager is None:
        import yaml
        cfg_path = ROOT / "training" / "configs" / "train_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        threshold = cfg["inference"]["confidence_threshold"]
        _flag_manager = FlagManager(threshold=threshold)
    return _flag_manager


# ── GET /health ───────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns 200 OK when the service is running."""
    return HealthResponse(status="ok", message="Service is running")


# ── POST /predict ─────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: QueryRequest):
    """
    Classify a customer support query.

    - **intent**   : complaint | inquiry | feedback
    - **priority** : high | medium | low
    - **flagged**  : True if either confidence score < threshold (default 0.70)
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query text must not be empty.")

    logger.info("Received prediction request", extra={"query_length": len(query)})

    try:
        predictor    = get_predictor()
        flag_manager = get_flag_manager()
        result       = predictor.predict(query)
    except Exception as exc:
        logger.error("Prediction failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    # Confidence-based flagging
    flag_manager.check_and_log(query, result)

    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Prediction complete",
        extra={
            "intent":              result["intent"],
            "priority":            result["priority"],
            "intent_confidence":   result["intent_confidence"],
            "priority_confidence": result["priority_confidence"],
            "flagged":             result["flagged"],
        },
    )

    return PredictionResponse(
        query=query,
        timestamp=timestamp,
        **result,
    )


# ── SageMaker Required Endpoints ──────────────────────────────────────────────
@router.get("/ping", tags=["Health"])
def ping():
    """SageMaker health check endpoint. Maps to /health."""
    return health_check()


@router.post("/invocations", response_model=PredictionResponse, tags=["Prediction"])
def invocations(request: QueryRequest):
    """SageMaker inference endpoint. Maps to /predict."""
    return predict(request)
