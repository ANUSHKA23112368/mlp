from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from civiclens.summarizer import summarize_text

URGENCY_SCORE = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SLA_TARGET_HOURS = {"low": 72, "medium": 24, "high": 8, "critical": 2}
IMPACT_LEXICON = {
    "public safety": ["unsafe", "harassment", "police", "crime", "emergency", "school"],
    "infrastructure": ["road", "pothole", "street", "light", "transformer", "manhole"],
    "water and sanitation": ["water", "leak", "overflow", "garbage", "sewer", "waste"],
    "health service": ["hospital", "clinic", "patient", "treatment", "ambulance"],
    "digital service": ["portal", "website", "app", "system", "online", "billing"],
}


def format_department(label: str) -> str:
    return label.replace("_", " ").title()


def top_keywords(texts: list[str], limit: int = 10) -> list[tuple[str, int]]:
    if not texts:
        return []
    try:
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)
        matrix = vectorizer.fit_transform(texts)
        counts = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        pairs = sorted(zip(terms, counts), key=lambda item: item[1], reverse=True)
        return [(term, int(count)) for term, count in pairs[:limit]]
    except ValueError:
        return []


def infer_impact_area(text: str) -> str:
    lowered = text.lower()
    best_area = "service quality"
    best_score = 0
    for area, keywords in IMPACT_LEXICON.items():
        score = sum(keyword in lowered for keyword in keywords)
        if score > best_score:
            best_area = area
            best_score = score
    return best_area


def confidence_band(score: float) -> str:
    if score >= 0.8:
        return "high confidence"
    if score >= 0.6:
        return "moderate confidence"
    return "needs analyst review"


def format_sla(hours: int) -> str:
    if hours < 24:
        return f"within {hours}h"
    days = hours // 24
    return f"within {days}d"


def recommend_action(department: str, urgency: str) -> str:
    team = format_department(department)
    if urgency == "critical":
        return f"Escalate to {team}, notify leadership, and assign a live owner immediately."
    if urgency == "high":
        return f"Route to {team} today and begin tracked response within the active shift."
    if urgency == "medium":
        return f"Queue for {team}, acknowledge receipt, and review during the next service cycle."
    return f"Bundle with the {team} backlog, monitor recurrence, and respond in the routine queue."


def add_risk_score(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["urgency_score"] = enriched["predicted_urgency"].map(URGENCY_SCORE).fillna(1)
    enriched["risk_score"] = (
        enriched["urgency_score"] * 24
        + enriched["complaint_text"].str.len().clip(upper=260) / 11
        + (1 - enriched["combined_confidence"]) * 18
    ).round(2)
    return enriched


def enrich_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "department_confidence" not in enriched.columns:
        enriched["department_confidence"] = 0.5
    if "urgency_confidence" not in enriched.columns:
        enriched["urgency_confidence"] = 0.5

    enriched["department_confidence"] = enriched["department_confidence"].astype(float)
    enriched["urgency_confidence"] = enriched["urgency_confidence"].astype(float)
    enriched["combined_confidence"] = (
        (enriched["department_confidence"] + enriched["urgency_confidence"]) / 2
    ).round(3)
    enriched["confidence_band"] = enriched["combined_confidence"].apply(confidence_band)
    enriched["impact_area"] = enriched["complaint_text"].apply(infer_impact_area)
    enriched["target_sla_hours"] = (
        enriched["predicted_urgency"].map(SLA_TARGET_HOURS).fillna(48).astype(int)
    )
    enriched["target_sla"] = enriched["target_sla_hours"].apply(format_sla)
    enriched["priority_label"] = enriched["predicted_urgency"].map(
        {
            "critical": "Immediate escalation",
            "high": "Rapid response",
            "medium": "Active monitoring",
            "low": "Routine queue",
        }
    ).fillna("Routine queue")
    enriched["recommended_action"] = enriched.apply(
        lambda row: recommend_action(row["predicted_department"], row["predicted_urgency"]),
        axis=1,
    )
    return add_risk_score(enriched)


def build_executive_report(frame: pd.DataFrame) -> str:
    overview = build_overview(frame)
    keywords = ", ".join(term for term, _ in overview["keywords"][:5]) or "no dominant repeat signals"
    return (
        "CivicLens AI Executive Brief\n\n"
        f"Total cases reviewed: {overview['total_cases']}\n"
        f"High-risk cases: {overview['high_risk_cases']}\n"
        f"Critical cases: {overview['critical_cases']}\n"
        f"Urgent share: {overview['urgent_rate']}%\n"
        f"Average confidence: {overview['average_confidence']}%\n"
        f"Most pressured department: {overview['most_loaded_department']}\n"
        f"Recurring issue signals: {keywords}\n\n"
        f"Leadership summary:\n{overview['executive_summary']}\n"
    )


def build_overview(frame: pd.DataFrame) -> dict[str, Any]:
    enriched = enrich_predictions(frame)
    by_department = (
        enriched.groupby("predicted_department")
        .agg(
            cases=("complaint_text", "size"),
            urgent_cases=(
                "predicted_urgency",
                lambda values: int(values.isin(["high", "critical"]).sum()),
            ),
            avg_risk_score=("risk_score", "mean"),
        )
        .sort_values(["urgent_cases", "cases"], ascending=False)
        .reset_index()
    )
    by_department["avg_risk_score"] = by_department["avg_risk_score"].round(2)
    urgency_distribution = (
        enriched.groupby("predicted_urgency")
        .size()
        .rename("cases")
        .reset_index()
    )
    urgency_distribution["sort_order"] = urgency_distribution["predicted_urgency"].map(
        {"critical": 0, "high": 1, "medium": 2, "low": 3}
    )
    urgency_distribution = urgency_distribution.sort_values("sort_order").drop(columns="sort_order")
    urgent_cases = enriched[enriched["predicted_urgency"].isin(["high", "critical"])]
    summary_source = " ".join(enriched["complaint_text"].head(12).tolist())
    executive_summary = summarize_text(summary_source, max_sentences=3)
    keywords = top_keywords(enriched["complaint_text"].tolist())
    action_board = (
        urgent_cases.groupby("predicted_department")
        .agg(
            urgent_cases=("complaint_text", "size"),
            avg_risk_score=("risk_score", "mean"),
            avg_sla_hours=("target_sla_hours", "mean"),
        )
        .sort_values(["urgent_cases", "avg_risk_score"], ascending=False)
        .reset_index()
    )
    if not action_board.empty:
        action_board["avg_risk_score"] = action_board["avg_risk_score"].round(2)
        action_board["avg_sla_hours"] = action_board["avg_sla_hours"].round(1)

    average_confidence = round(enriched["combined_confidence"].mean() * 100, 1)
    most_loaded_department = (
        format_department(by_department.iloc[0]["predicted_department"])
        if not by_department.empty
        else "No data"
    )

    return {
        "total_cases": int(len(enriched)),
        "high_risk_cases": int(len(urgent_cases)),
        "critical_cases": int((enriched["predicted_urgency"] == "critical").sum()),
        "urgent_rate": round((len(urgent_cases) / len(enriched)) * 100, 1),
        "average_risk_score": float(enriched["risk_score"].mean().round(2)),
        "average_confidence": average_confidence,
        "most_loaded_department": most_loaded_department,
        "department_distribution": by_department.to_dict(orient="records"),
        "urgency_distribution": urgency_distribution.to_dict(orient="records"),
        "action_board": action_board.to_dict(orient="records"),
        "keywords": keywords,
        "executive_summary": executive_summary,
        "top_risks": enriched.sort_values("risk_score", ascending=False)
        .head(6)[
            [
                "complaint_text",
                "predicted_department",
                "predicted_urgency",
                "combined_confidence",
                "impact_area",
                "target_sla",
                "priority_label",
                "risk_score",
            ]
        ]
        .to_dict(orient="records"),
    }
