from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import build_overview
from civiclens.data import load_demo_data
from civiclens.modeling import train_models
from civiclens.summarizer import summarize_text

app = FastAPI(title="CivicLens AI API", version="1.0.0")
artifacts = train_models(load_demo_data())


class ComplaintRequest(BaseModel):
    complaint_text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ComplaintRequest) -> dict[str, str]:
    prediction = artifacts.predict([request.complaint_text]).iloc[0]
    return {
        "predicted_department": prediction["predicted_department"],
        "predicted_urgency": prediction["predicted_urgency"],
        "summary": summarize_text(request.complaint_text, max_sentences=2),
    }


@app.get("/overview")
def overview() -> dict:
    predictions = artifacts.predict(load_demo_data()["complaint_text"].tolist())
    return build_overview(predictions)

