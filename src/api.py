#!/usr/bin/env python3
"""
FastAPI REST API for the Sentiment Analysis engine.
Run with: uvicorn src.api:app --reload --port 8000
"""

import os
import sys
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="SOTA multi-backend sentiment analysis: VADER + ML + Transformer",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Lazy-loaded engine singleton
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from src.sentiment_engine import SentimentEngine
        _engine = SentimentEngine(
            use_ml_model=True,
            use_transformer=os.getenv("USE_TRANSFORMER", "0") in ("1", "true", "yes"),
        )
    return _engine


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50, description="List of texts")


@app.get("/health")
def health():
    engine = get_engine()
    return {"status": "ok", "backends": engine.available_backends}


@app.post("/predict")
def predict(req: TextRequest):
    """Single text analysis — returns full enriched result."""
    try:
        result = get_engine().analyze(req.text)
        return result
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/batch")
def batch_predict(req: BatchRequest):
    """Batch analysis — returns list of results."""
    try:
        results = get_engine().analyze_batch(req.texts)
        return {"results": results, "count": len(results)}
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc))
