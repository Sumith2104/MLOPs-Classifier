"""
query_schema.py
----------------
Pydantic models for the /predict API endpoint.

Request  → QueryRequest
Response → PredictionResponse
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Optional


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
    intent:              Optional[str]   = Field(default=None, description="complaint | inquiry | feedback")
    priority:            Optional[str]   = Field(default=None, description="high | medium | low")
    intent_confidence:   Optional[float] = Field(default=None, description="Softmax confidence for intent (0–1)")
    priority_confidence: Optional[float] = Field(default=None, description="Softmax confidence for priority (0–1)")
    flagged:             Optional[bool]  = Field(default=None, description="True if either confidence < threshold")
    timestamp:           str   = Field(description="ISO 8601 prediction timestamp")
    error:               Optional[str]   = Field(default=None, description="Error message if prediction failed")


class HealthResponse(BaseModel):
    status:  str = "ok"
    message: str = "Service is running"


class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(
        ...,
        description="List of raw customer support query texts",
        examples=[["My order is damaged.", "Where is my package?", "Excellent service!"]],
    )

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Query list cannot be empty")
        if len(v) > 50:
            raise ValueError("Query list cannot exceed 50 items")
        
        seen = set()
        for idx, q in enumerate(v):
            if not isinstance(q, str):
                raise ValueError(f"Query at index {idx} must be a string")
            stripped = q.strip()
            if not stripped:
                raise ValueError(f"Query at index {idx} is empty or blank")
            if len(stripped) < 3:
                raise ValueError(f"Query at index {idx} must be at least 3 characters long")
            if len(stripped) > 512:
                raise ValueError(f"Query at index {idx} must be at most 512 characters long")
            if stripped in seen:
                raise ValueError(f"Duplicate query found: '{stripped}'")
            seen.add(stripped)
        return v


class BatchPredictionResponse(BaseModel):
    results:             List[PredictionResponse] = Field(description="Individual prediction results")
    total:               int = Field(description="Total number of queries in the batch")
    flagged_count:       int = Field(description="Number of queries flagged for human review")
    timestamp:           str = Field(description="ISO 8601 prediction timestamp")
    processing_time_ms:  float = Field(description="Time taken to process the batch in milliseconds")

