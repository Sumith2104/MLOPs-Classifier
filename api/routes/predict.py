"""
predict.py
-----------
FastAPI route for POST /predict and GET /health.
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.schemas.query_schema import QueryRequest, PredictionResponse, HealthResponse, BatchQueryRequest, BatchPredictionResponse
from model.predictor import Predictor
from monitoring.logger import get_logger
from api.database import get_supabase
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


def warm_up_serving() -> None:
    """Warm up the model and flag manager at startup to avoid first-request latency."""
    logger.info("Warming up prediction service...")
    try:
        get_predictor()
        get_flag_manager()
        logger.info("Prediction service warm-up complete.")
    except Exception as exc:
        logger.error(f"Prediction service warm-up failed: {exc}")


_batch_semaphore: asyncio.Semaphore | None = None

def get_batch_semaphore() -> asyncio.Semaphore:
    global _batch_semaphore
    if _batch_semaphore is None:
        import yaml
        cfg_path = ROOT / "training" / "configs" / "train_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        limit = cfg.get("serving", {}).get("max_concurrent_batches", 3)
        _batch_semaphore = asyncio.Semaphore(limit)
    return _batch_semaphore


def get_timeout_seconds() -> float:
    import yaml
    cfg_path = ROOT / "training" / "configs" / "train_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return float(cfg.get("serving", {}).get("timeout_seconds", 25.0))


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

    db = get_supabase()
    if db:
        try:
            db.table("classified_queries").insert({
                "message": query,
                "intent": result["intent"],
                "priority": result["priority"],
                "intent_confidence": result["intent_confidence"],
                "priority_confidence": result["priority_confidence"],
                "flagged": result["flagged"]
            }).execute()
        except Exception as exc:
            logger.error("Failed to log to Supabase", extra={"error": str(exc)})


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


# ── POST /predict/batch ───────────────────────────────────────────────────────
@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchQueryRequest):
    """
    Classify a batch of customer support queries.
    Supports up to 50 queries in a single call.
    Uses concurrency limit and timeout guard.
    """
    queries = request.queries
    
    semaphore = get_batch_semaphore()
    if semaphore.locked():
        logger.warning("Batch prediction queue full, rejecting request.")
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent batch requests. Please try again later."
        )
        
    start_time = time.time()
    logger.info("Received batch prediction request", extra={"batch_size": len(queries)})
    
    async with semaphore:
        try:
            predictor = get_predictor()
            timeout_seconds = get_timeout_seconds()
            
            # Execute in a thread pool to avoid blocking the event loop
            results = await asyncio.wait_for(
                asyncio.to_thread(predictor.predict_batch, queries),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error("Batch prediction request timed out")
            raise HTTPException(status_code=504, detail="Batch prediction request timed out.")
        except Exception as exc:
            logger.error("Batch prediction failed", extra={"error": str(exc)})
            raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")
            
    # Process results, flagging and preparing response
    flag_manager = get_flag_manager()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    prediction_responses = []
    flagged_count = 0
    db_records = []
    
    for query, res in zip(queries, results):
        if "error" in res and res["error"] is not None:
            response_item = PredictionResponse(
                query=query,
                timestamp=timestamp,
                error=res["error"],
                intent=None,
                priority=None,
                intent_confidence=None,
                priority_confidence=None,
                flagged=True
            )
            flagged_count += 1
        else:
            is_flagged = flag_manager.check_and_log(query, res)
            if is_flagged:
                flagged_count += 1
                
            response_item = PredictionResponse(
                query=query,
                timestamp=timestamp,
                **res
            )
            
            db_records.append({
                "message": query,
                "intent": res["intent"],
                "priority": res["priority"],
                "intent_confidence": res["intent_confidence"],
                "priority_confidence": res["priority_confidence"],
                "flagged": res["flagged"]
            })
            
        prediction_responses.append(response_item)
        
    # Bulk write to database
    if db_records:
        db = get_supabase()
        if db:
            try:
                db.table("classified_queries").insert(db_records).execute()
            except Exception as exc:
                logger.error("Failed to bulk log to Supabase", extra={"error": str(exc)})
                
    processing_time_ms = (time.time() - start_time) * 1000
    
    logger.info(
        "Batch prediction complete",
        extra={
            "total_queries": len(queries),
            "flagged_count": flagged_count,
            "processing_time_ms": round(processing_time_ms, 2)
        }
    )
    
    return BatchPredictionResponse(
        results=prediction_responses,
        total=len(queries),
        flagged_count=flagged_count,
        timestamp=timestamp,
        processing_time_ms=round(processing_time_ms, 2)
    )
