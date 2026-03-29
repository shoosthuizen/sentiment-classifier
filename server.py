"""
FastAPI inference server for the fine-tuned DistilBERT sentiment classifier.

Endpoints
---------
POST /predict          — classify a single text
POST /predict/batch    — classify a list of texts
GET  /health           — liveness check
"""

import os
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "model/best")
MAX_LENGTH = 128


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Sentiment Classifier",
    description="Binary sentiment analysis (positive / negative) with DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048, example="I loved this movie!")


class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32, example=["Great!", "Terrible."])


class Prediction(BaseModel):
    text: str
    label: str
    score: float = Field(..., description="Confidence in the predicted label (0–1)")


class BatchPrediction(BaseModel):
    predictions: List[Prediction]


# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------

tokenizer = None
model = None
device = None


def load_model():
    global tokenizer, model, device

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"Model directory '{MODEL_DIR}' not found. "
            "Run train.py first, or set the MODEL_DIR environment variable."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_DIR} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("Model ready.")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _predict(texts: List[str]) -> List[Prediction]:
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1).tolist()
    pred_probs = probs.max(dim=-1).values.tolist()

    return [
        Prediction(
            text=text,
            label=model.config.id2label[pred_id],
            score=round(prob, 4),
        )
        for text, pred_id, prob in zip(texts, pred_ids, pred_probs)
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model_dir": MODEL_DIR}


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(body: TextInput):
    """Classify a single piece of text."""
    try:
        return _predict([body.text])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPrediction, tags=["inference"])
def predict_batch(body: BatchInput):
    """Classify up to 32 texts in one call."""
    try:
        return BatchPrediction(predictions=_predict(body.texts))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
