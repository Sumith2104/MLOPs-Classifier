"""
query_schema.py
----------------
Pydantic models for the /predict API endpoint.

Request  → QueryRequest
Response → PredictionResponse
"""

from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Raw customer support query text",
        examples=["My order never arrived and it has been 2 weeks!"],
    )


class PredictionResponse(BaseModel):
    query:               str
    intent:              str   = Field(description="complaint | inquiry | feedback")
    priority:            str   = Field(description="high | medium | low")
    intent_confidence:   float = Field(description="Softmax confidence for intent (0–1)")
    priority_confidence: float = Field(description="Softmax confidence for priority (0–1)")
    flagged:             bool  = Field(description="True if either confidence < threshold")
    timestamp:           str   = Field(description="ISO 8601 prediction timestamp")


class HealthResponse(BaseModel):
    status:  str = "ok"
    message: str = "Service is running"
