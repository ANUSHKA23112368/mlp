# CivicShield AI API Reference

This document summarizes the core REST endpoints provided by the CivicShield AI FastAPI service (`api.py`).

## Base URL
- Local Dev: `http://127.0.0.1:8000`
- Swagger Interactive Docs: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

---

## 1. System Endpoints

### `GET /health`
Returns the operational health and model accuracy diagnostics.
- **Response**:
```json
{
  "status": "ok",
  "version": "3.0.0",
  "active_algorithm": "logistic_regression",
  "sample_count": 150,
  "department_accuracy": 0.88,
  "urgency_accuracy": 0.85
}
```

---

## 2. Civic Intelligence & Inference

### `POST /predict`
Analyzes a single citizen complaint text and returns department classification, SLA, frustration index, and XAI token attributions.
- **Payload**:
```json
{
  "complaint_text": "Water pipeline burst near the market, causing street flooding."
}
```

### `POST /predict/batch`
Batch processes multiple complaints concurrently.

### `POST /explain`
Returns token-level attribution and highlighted HTML for model explainability.

### `POST /deduplicate`
Identifies near-duplicate complaints using cosine similarity and returns incident clusters.

---

## 3. Security Operations

### `POST /security/scan`
Executes security threat detection across rule engine, ML anomaly detector (Isolation Forest), and incident thread correlator.
- **Payload**:
```json
{
  "event_count": 50,
  "hours_back": 24,
  "anomaly_contamination": 0.08
}
```

### `GET /security/summary`
Returns high-level security event category distribution.
