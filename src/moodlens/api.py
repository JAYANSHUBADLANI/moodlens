"""FastAPI inference service.

Run:  uvicorn moodlens.api:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import LABELS
from .predict import get_model, predict

app = FastAPI(
    title="MoodLens API",
    description="Emotion detection for short text (6 classes: "
    + ", ".join(LABELS)
    + ").",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["i am so happy today"])
    top_k: int = Field(3, ge=1, le=6)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    top_k: int = Field(3, ge=1, le=6)


@app.get("/health")
def health() -> dict:
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}


@app.get("/labels")
def labels() -> dict:
    return {"labels": LABELS}


@app.post("/predict")
def predict_one(req: PredictRequest) -> dict:
    try:
        return predict(req.text, top_k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch")
def predict_many(req: BatchRequest) -> dict:
    try:
        return {"results": [predict(t, top_k=req.top_k) for t in req.texts if t.strip()]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
