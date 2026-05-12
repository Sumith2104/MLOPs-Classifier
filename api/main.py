"""
main.py
--------
FastAPI application entrypoint.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.predict import router as predict_router

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
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict_router, prefix="")


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
