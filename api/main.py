"""
main.py
--------
FastAPI application entrypoint.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.predict import router as predict_router
from api.routes.mlops import router as mlops_router
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.routes.predict import warm_up_serving
from api.worker import start_query_classifier_worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    warm_up_serving()
    worker_task = asyncio.create_task(start_query_classifier_worker())
    yield
    # Shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Customer Support Query Classifier",
    description=(
        "AI-powered REST API that classifies customer support queries "
        "into Intent (complaint / inquiry / feedback) and "
        "Priority (high / medium / low) using a fine-tuned BERT model."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
app.add_middleware(RateLimiterMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict_router, prefix="")
app.include_router(mlops_router, prefix="")


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Customer Support Query Classifier",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        },
    }
