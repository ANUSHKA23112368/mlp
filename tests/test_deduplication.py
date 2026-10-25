from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.deduplication import (
    calculate_deduplication_metrics,
    find_incident_clusters,
    find_similar_complaints,
)


def test_find_similar_complaints():
    database = [
        "Major water pipeline burst on 5th Avenue near the central library.",
        "Broken traffic light at Central Crossway causing heavy traffic jams.",
        "Transformer caught fire on Industrial Road near gate 3.",
    ]
    query = "Water pipe ruptured and leaking water on 5th Avenue"
    matches = find_similar_complaints(query, database, top_k=2, threshold=0.15)
    assert len(matches) >= 1
    assert "5th Avenue" in matches[0]["complaint_text"]
    assert matches[0]["similarity_score"] > 0.15


def test_find_incident_clusters():
    complaints = [
        "Electric transformer exploded near gate 3, smoke billowing.",
        "Transformer caught fire near gate 3, sparking dangerously.",
        "Complete power failure after transformer sparked near gate 3.",
        "The bus route 42 was delayed by 30 minutes this morning.",
    ]
    clusters = find_incident_clusters(complaints, similarity_threshold=0.20, min_cluster_size=2)
    assert len(clusters) >= 1
    top_cluster = clusters[0]
    assert top_cluster["total_reports"] >= 2
    assert "transformer" in top_cluster["signature_keywords"] or "gate" in top_cluster["signature_keywords"]

    metrics = calculate_deduplication_metrics(clusters, len(complaints))
    assert metrics["duplicate_cases_found"] >= 1
    assert metrics["deduplication_rate_pct"] > 0
