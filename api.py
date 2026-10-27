from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ── CivicShield extensions ─────────────────────────────────────────────────
from civicshield.shared.logging import configure_logging
from civicshield.shared.middleware import register_middleware
from civicshield.auth.router import router as auth_router

configure_logging()

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import (
    build_executive_report,
    build_geo_and_channel_overview,
    build_overview,
    enrich_predictions,
    top_keywords,
)
from civiclens.data import load_demo_data
from civiclens.deduplication import calculate_deduplication_metrics, find_incident_clusters, find_similar_complaints
from civiclens.explainability import explain_prediction
from civiclens.modeling import benchmark_models, train_models
from civiclens.summarizer import summarize_text
from civiclens.topics import discover_topics

app = FastAPI(
    title="CivicShield AI API",
    version="3.0.0",
    description="Intelligent Civic Operations and Cybersecurity Platform.",
)

# Middleware and routers
register_middleware(app)
app.include_router(auth_router)

# Global dataset and trained model artifacts
demo_data = load_demo_data()
artifacts = train_models(demo_data)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------
class ComplaintRequest(BaseModel):
    complaint_text: str = Field(
        ...,
        min_length=5,
        description="Raw narrative of the citizen complaint or customer ticket.",
        examples=["The water main on 5th Avenue ruptured and water is entering shop basements."],
    )


class BatchComplaintRequest(BaseModel):
    complaints: list[str] = Field(
        ...,
        min_length=1,
        description="List of raw complaint narratives to evaluate in batch.",
    )


class ExplainRequest(BaseModel):
    complaint_text: str = Field(..., min_length=5)
    target: str = Field("department", description="Target model to explain: 'department' or 'urgency'")
    top_k: int = Field(6, ge=1, le=20, description="Number of top explanatory features to return")


class DeduplicationRequest(BaseModel):
    complaints: list[str] | None = Field(
        default=None,
        description="Optional list of complaint texts to cluster. If omitted, active database is clustered.",
    )
    similarity_threshold: float = Field(0.24, ge=0.1, le=0.9, description="Cosine similarity cutoff threshold")
    min_cluster_size: int = Field(2, ge=2, description="Minimum reports required to form an incident cluster")


class SimilarQueryRequest(BaseModel):
    query_text: str = Field(..., min_length=5)
    top_k: int = Field(4, ge=1, le=10)
    threshold: float = Field(0.20, ge=0.05, le=0.95)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "active_algorithm": artifacts.algorithm,
        "sample_count": len(demo_data),
        "department_accuracy": artifacts.metrics.get("department_accuracy"),
        "urgency_accuracy": artifacts.metrics.get("urgency_accuracy"),
    }


@app.post("/predict", tags=["Inference"])
def predict(request: ComplaintRequest) -> dict[str, Any]:
    prediction = enrich_predictions(artifacts.predict([request.complaint_text])).iloc[0]
    
    # Feature attributions for department and urgency
    dept_exp = explain_prediction(artifacts.department_model, request.complaint_text, top_k=4)
    urg_exp = explain_prediction(artifacts.urgency_model, request.complaint_text, top_k=4)

    return {
        "predicted_department": prediction["predicted_department"],
        "predicted_urgency": prediction["predicted_urgency"],
        "department_confidence": float(prediction["department_confidence"]),
        "urgency_confidence": float(prediction["urgency_confidence"]),
        "combined_confidence": float(prediction["combined_confidence"]),
        "confidence_band": prediction["confidence_band"],
        "priority_label": prediction["priority_label"],
        "impact_area": prediction["impact_area"],
        "target_sla": prediction["target_sla"],
        "target_sla_hours": int(prediction["target_sla_hours"]),
        "risk_score": float(prediction["risk_score"]),
        "recommended_action": prediction["recommended_action"],
        "summary": summarize_text(request.complaint_text, max_sentences=2),
        "issue_signals": [term for term, _ in top_keywords([request.complaint_text], limit=5)],
        # Advanced Sentiment & Frustration Intelligence
        "sentiment_score": float(prediction["sentiment_score"]),
        "frustration_score": int(prediction["frustration_score"]),
        "escalation_risk": prediction["escalation_risk"],
        "frustration_signals": prediction["frustration_signals"],
        "liaison_guidance": prediction["liaison_guidance"],
        # Explainable AI Attribution
        "department_key_tokens": dept_exp["positive_features"],
        "urgency_key_tokens": urg_exp["positive_features"],
    }


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(request: BatchComplaintRequest) -> dict[str, Any]:
    raw_df = artifacts.predict(request.complaints)
    enriched = enrich_predictions(raw_df)
    
    records = []
    for _, row in enriched.iterrows():
        records.append({
            "complaint_text": row["complaint_text"],
            "predicted_department": row["predicted_department"],
            "predicted_urgency": row["predicted_urgency"],
            "combined_confidence": float(row["combined_confidence"]),
            "priority_label": row["priority_label"],
            "target_sla": row["target_sla"],
            "risk_score": float(row["risk_score"]),
            "frustration_score": int(row["frustration_score"]),
            "escalation_risk": row["escalation_risk"],
            "recommended_action": row["recommended_action"],
        })

    return {
        "total_processed": len(records),
        "results": records,
    }


@app.post("/explain", tags=["Explainability (XAI)"])
def explain(request: ExplainRequest) -> dict[str, Any]:
    target_lower = request.target.lower().strip()
    if target_lower in ["dept", "department"]:
        pipeline = artifacts.department_model
    elif target_lower in ["urg", "urgency"]:
        pipeline = artifacts.urgency_model
    else:
        raise HTTPException(status_code=400, detail="Target must be 'department' or 'urgency'")

    return explain_prediction(pipeline, request.complaint_text, top_k=request.top_k)


@app.post("/deduplicate", tags=["Operations & Clustering"])
def deduplicate(request: DeduplicationRequest) -> dict[str, Any]:
    complaint_list = request.complaints if request.complaints is not None else demo_data["complaint_text"].tolist()
    clusters = find_incident_clusters(
        complaint_list,
        similarity_threshold=request.similarity_threshold,
        min_cluster_size=request.min_cluster_size,
    )
    metrics = calculate_deduplication_metrics(clusters, len(complaint_list))
    return {
        "metrics": metrics,
        "clusters": clusters,
    }


@app.post("/similar", tags=["Operations & Clustering"])
def find_similar(request: SimilarQueryRequest) -> dict[str, Any]:
    db_texts = demo_data["complaint_text"].tolist()
    matches = find_similar_complaints(
        query_text=request.query_text,
        database_texts=db_texts,
        top_k=request.top_k,
        threshold=request.threshold,
    )
    return {
        "query_text": request.query_text,
        "matches_found": len(matches),
        "matches": matches,
    }


@app.get("/topics", tags=["Unsupervised NLP"])
def get_topics(n_topics: int = Query(5, ge=2, le=10)) -> dict[str, Any]:
    topics = discover_topics(demo_data["complaint_text"].tolist(), n_topics=n_topics)
    return {
        "total_topics": len(topics),
        "topics": topics,
    }


@app.get("/benchmark", tags=["Model Quality"])
def benchmark(cv_folds: int = Query(3, ge=2, le=5)) -> dict[str, Any]:
    return benchmark_models(demo_data, cv_splits=cv_folds)


@app.get("/geo-analytics", tags=["Analytics"])
def geo_analytics() -> dict[str, Any]:
    preds = artifacts.predict(demo_data["complaint_text"].tolist())
    for col in ["ward", "channel", "timestamp", "status"]:
        if col in demo_data.columns:
            preds[col] = demo_data[col]
    return build_geo_and_channel_overview(preds)


@app.get("/overview", tags=["Analytics"])
def overview() -> dict[str, Any]:
    predictions = artifacts.predict(demo_data["complaint_text"].tolist())
    for col in ["ward", "channel", "timestamp", "status"]:
        if col in demo_data.columns:
            predictions[col] = demo_data[col]
    return build_overview(predictions)


@app.get("/brief", tags=["Analytics"])
def brief() -> dict[str, str]:
    predictions = artifacts.predict(demo_data["complaint_text"].tolist())
    return {"report": build_executive_report(predictions)}
