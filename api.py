from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import build_executive_report, build_overview, enrich_predictions, top_keywords
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
def predict(request: ComplaintRequest) -> dict[str, str | float | list[str]]:
    prediction = enrich_predictions(artifacts.predict([request.complaint_text])).iloc[0]
    return {
        "predicted_department": prediction["predicted_department"],
        "predicted_urgency": prediction["predicted_urgency"],
        "department_confidence": float(prediction["department_confidence"]),
        "urgency_confidence": float(prediction["urgency_confidence"]),
        "combined_confidence": float(prediction["combined_confidence"]),
        "priority_label": prediction["priority_label"],
        "impact_area": prediction["impact_area"],
        "target_sla": prediction["target_sla"],
        "recommended_action": prediction["recommended_action"],
        "summary": summarize_text(request.complaint_text, max_sentences=2),
        "issue_signals": [term for term, _ in top_keywords([request.complaint_text], limit=5)],
    }


@app.get("/overview")
def overview() -> dict:
    predictions = artifacts.predict(load_demo_data()["complaint_text"].tolist())
    return build_overview(predictions)


@app.get("/brief")
def brief() -> dict[str, str]:
    predictions = artifacts.predict(load_demo_data()["complaint_text"].tolist())
    return {"report": build_executive_report(predictions)}
