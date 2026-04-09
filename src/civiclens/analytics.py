from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from civiclens.summarizer import summarize_text

URGENCY_SCORE = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def top_keywords(texts: list[str], limit: int = 10) -> list[tuple[str, int]]:
    if not texts:
        return []
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)
    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    pairs = sorted(zip(terms, counts), key=lambda item: item[1], reverse=True)
    return [(term, int(count)) for term, count in pairs[:limit]]


def add_risk_score(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["urgency_score"] = enriched["predicted_urgency"].map(URGENCY_SCORE).fillna(1)
    enriched["risk_score"] = (
        enriched["urgency_score"] * 25
        + enriched["complaint_text"].str.len().clip(upper=250) / 10
    ).round(2)
    return enriched


def build_overview(frame: pd.DataFrame) -> dict[str, Any]:
    enriched = add_risk_score(frame)
    by_department = (
        enriched.groupby("predicted_department")
        .size()
        .sort_values(ascending=False)
        .rename("cases")
        .reset_index()
    )
    urgent_cases = enriched[enriched["predicted_urgency"].isin(["high", "critical"])]
    summary_source = " ".join(enriched["complaint_text"].head(12).tolist())
    executive_summary = summarize_text(summary_source, max_sentences=3)
    keywords = top_keywords(enriched["complaint_text"].tolist())

    return {
        "total_cases": int(len(enriched)),
        "high_risk_cases": int(len(urgent_cases)),
        "average_risk_score": float(enriched["risk_score"].mean().round(2)),
        "department_distribution": by_department.to_dict(orient="records"),
        "keywords": keywords,
        "executive_summary": executive_summary,
        "top_risks": enriched.sort_values("risk_score", ascending=False)
        .head(5)[
            [
                "complaint_text",
                "predicted_department",
                "predicted_urgency",
                "risk_score",
            ]
        ]
        .to_dict(orient="records"),
    }

