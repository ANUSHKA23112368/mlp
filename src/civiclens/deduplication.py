from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_complaints(
    query_text: str,
    database_texts: list[str],
    top_k: int = 4,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    if not database_texts or not query_text.strip():
        return []

    all_texts = [query_text] + database_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vec = tfidf_matrix[0]
    db_matrix = tfidf_matrix[1:]

    sim_scores = cosine_similarity(query_vec, db_matrix).flatten()
    sorted_indices = np.argsort(sim_scores)[::-1]

    results = []
    for idx in sorted_indices:
        score = float(sim_scores[idx])
        if score < threshold or len(results) >= top_k:
            break
        results.append({
            "complaint_text": database_texts[idx],
            "similarity_score": round(score, 3),
            "match_rank": len(results) + 1,
        })

    return results


def find_incident_clusters(
    complaints: list[str] | pd.DataFrame,
    similarity_threshold: float = 0.22,
    min_cluster_size: int = 2,
) -> list[dict[str, Any]]:
    if isinstance(complaints, pd.DataFrame):
        texts = complaints["complaint_text"].tolist()
        has_df = True
    else:
        texts = list(complaints)
        has_df = False

    if len(texts) < min_cluster_size:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    n = len(texts)
    visited = set()
    clusters: list[dict[str, Any]] = []

    # Connected components / Greedy clique formation
    for i in range(n):
        if i in visited:
            continue

        sim_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]
        # Exclude already visited
        unvisited_sim = [int(idx) for idx in sim_indices if idx not in visited]

        if len(unvisited_sim) >= min_cluster_size:
            for idx in unvisited_sim:
                visited.add(idx)

            # Pick the primary incident: longest or earliest complaint text
            cluster_texts = [texts[idx] for idx in unvisited_sim]
            primary_idx = max(unvisited_sim, key=lambda idx: len(texts[idx]))
            primary_text = texts[primary_idx]
            corroborating = [texts[idx] for idx in unvisited_sim if idx != primary_idx]

            # Extract common signature keywords
            cluster_tfidf = tfidf_matrix[unvisited_sim].mean(axis=0).A1
            feature_names = vectorizer.get_feature_names_out()
            top_term_indices = np.argsort(cluster_tfidf)[::-1][:4]
            cluster_signature = [str(feature_names[ti]) for ti in top_term_indices if cluster_tfidf[ti] > 0]

            cluster_item: dict[str, Any] = {
                "cluster_id": len(clusters) + 1,
                "incident_topic": " / ".join(cluster_signature[:3]).title(),
                "primary_complaint": primary_text,
                "corroborating_count": len(corroborating),
                "total_reports": len(unvisited_sim),
                "indices": unvisited_sim,
                "signature_keywords": cluster_signature,
                "corroborating_reports": corroborating,
            }

            if has_df:
                sub_df = complaints.iloc[unvisited_sim]
                if "department" in sub_df.columns:
                    cluster_item["department"] = sub_df["department"].mode().iloc[0]
                elif "predicted_department" in sub_df.columns:
                    cluster_item["department"] = sub_df["predicted_department"].mode().iloc[0]

                if "urgency" in sub_df.columns:
                    cluster_item["urgency"] = sub_df["urgency"].mode().iloc[0]
                elif "predicted_urgency" in sub_df.columns:
                    cluster_item["urgency"] = sub_df["predicted_urgency"].mode().iloc[0]

                if "ward" in sub_df.columns:
                    cluster_item["ward"] = sub_df["ward"].mode().iloc[0]

            clusters.append(cluster_item)

    # Sort clusters by report count descending
    clusters.sort(key=lambda c: c["total_reports"], reverse=True)
    return clusters


def calculate_deduplication_metrics(
    clusters: list[dict[str, Any]], total_complaints: int
) -> dict[str, Any]:
    if total_complaints == 0:
        return {
            "duplicate_cases_found": 0,
            "deduplication_rate_pct": 0.0,
            "dispatches_saved": 0,
            "incident_clusters_count": 0,
        }

    duplicate_cases = sum(c["corroborating_count"] for c in clusters)
    rate = round((duplicate_cases / total_complaints) * 100, 1)

    return {
        "duplicate_cases_found": duplicate_cases,
        "deduplication_rate_pct": rate,
        "dispatches_saved": duplicate_cases,
        "incident_clusters_count": len(clusters),
    }
