from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"
    assert "sample_count" in data


def test_api_predict():
    res = client.post(
        "/predict",
        json={"complaint_text": "Drinking water supply pipeline burst on market road."},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_department"] == "water"
    assert "priority_label" in data
    assert "target_sla" in data
    assert "frustration_score" in data
    assert "escalation_risk" in data
    assert "department_key_tokens" in data


def test_api_predict_batch():
    res = client.post(
        "/predict/batch",
        json={
            "complaints": [
                "Pothole caused bike accident on highway.",
                "Hospital lost lab reports and delayed treatment.",
            ]
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2


def test_api_explain():
    res = client.post(
        "/explain",
        json={
            "complaint_text": "Streetlight timer broken and dark at night",
            "target": "department",
            "top_k": 4,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "positive_features" in data
    assert "highlighted_html" in data


def test_api_deduplicate():
    res = client.post("/deduplicate", json={"similarity_threshold": 0.20})
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "clusters" in data


def test_api_similar():
    res = client.post(
        "/similar",
        json={
            "query_text": "Water pipeline burst and leaking clean drinking water.",
            "top_k": 2,
            "threshold": 0.15,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "matches" in data


def test_api_topics():
    res = client.get("/topics?n_topics=4")
    assert res.status_code == 200
    data = res.json()
    assert data["total_topics"] == 4


def test_api_overview_and_brief():
    ov = client.get("/overview")
    assert ov.status_code == 200
    assert "total_cases" in ov.json()

    br = client.get("/brief")
    assert br.status_code == 200
    assert "CivicLens AI Executive Brief" in br.json()["report"]


def test_api_geo_analytics():
    geo = client.get("/geo-analytics")
    assert geo.status_code == 200
    data = geo.json()
    assert "ward_distribution" in data
    assert "channel_distribution" in data
