# CivicShield AI v3.0

**CivicShield AI** is an Intelligent Civic Operations & Cybersecurity Platform that combines citizen complaint intelligence with security event monitoring. It automates department routing, triage urgency, Explainable AI (XAI) feature attribution, near-duplicate incident clustering, citizen frustration tracking, security threat detection, anomaly monitoring, and executive reporting — all from a unified command centre.

> Evolved from CivicLens AI v2.0 — all original civic intelligence capabilities are preserved and enhanced.

---

## What makes this project standout

- **Dual Intelligence Platform**: Combines civic complaint ML classification with cybersecurity event monitoring in a single dashboard.
- **Security Operations Center**: Real-time security event ingestion, rule-based threat detection, Isolation Forest anomaly detection, and correlated threat threads.
- **Enhanced ML Pipeline**: Character n-gram features alongside word TF-IDF, data quality checks, model versioning, and enhanced metrics (macro/micro F1, precision, recall).
- **Pure ML & NLP Engineering**: Built with scikit-learn, TF-IDF vectorization, NMF topic decomposition, Isolation Forest anomaly detection, and calibrated models.
- **Explainable AI (XAI)**: Token-level local feature attribution showing exactly which words influenced department routing and urgency classifications.
- **Incident Deduplication Hub**: Groups corroborating citizen reports regarding the same underlying incident, measuring dispatches saved.
- **Citizen Frustration Index (0–100)**: Quantifies citizen emotional distress and flags escalation risk levels with recommended liaison guidance.
- **Multi-Model Benchmark Engine**: Compares Logistic Regression, Calibrated Linear SVM, Multinomial Naive Bayes, and Random Forest using Stratified K-Fold Cross-Validation.
- **JWT Authentication & RBAC**: Secure API access with role-based access control (admin, analyst, viewer) and structured JSON logging.
- **10-Tab Interactive Dashboard**: Streamlit dashboard with Command Center, Security Ops, ML Lab, Data Quality, and more.
- **Enterprise FastAPI Service**: Production REST API with auth, batch processing, explainability endpoints, middleware, and OpenAPI documentation.
- **52 Automated Tests**: Comprehensive test suite covering pipelines, API, auth, security module, ML enhancements, explainability, and deduplication.
- **CI/CD with GitHub Actions**: Automated testing on push, security scanning with Bandit, and secret detection with Gitleaks.

---

## Latest Updates

- **v3.0 Dark Theme Overhaul** *(Oct 2026)*: Complete dark theme redesign for Command Center and all dashboard tabs with improved contrast, modern glassmorphism styling, and enhanced readability.

---

## Architecture & Project Structure

```text
.
|-- app.py                              # Streamlit 10-tab interactive dashboard
|-- api.py                              # FastAPI REST service with auth & middleware
|-- pytest.ini                          # Test configuration
|-- requirements.txt                    # Python dependencies
|-- Dockerfile                          # Production container configuration
|-- docker-compose.yml                  # Local dev: API + PostgreSQL + Redis
|-- render.yaml                         # Cloud deployment configuration
|-- .env.example                        # Environment variable template
|-- .github/
|   `-- workflows/
|       |-- test.yml                    # CI: pytest on push/PR
|       `-- security.yml                # CI: Bandit SAST + Gitleaks
|-- data/
|   `-- demo_complaints.csv             # 150 realistic complaints with Ward, Channel, Status & Timestamp
|-- src/
|   `-- civiclens/                      # Original civic intelligence modules
|       |-- __init__.py
|       |-- analytics.py                # SLA calculations, risk scoring & geo/channel analytics
|       |-- data.py                     # Dataset validation, loading & default imputation
|       |-- deduplication.py            # Cosine similarity clustering & incident deduplication
|       |-- explainability.py           # XAI token attribution & HTML markup generation
|       |-- modeling.py                 # Multi-model training, benchmarking & persistence
|       |-- sentiment.py                # Citizen frustration scoring & escalation risk detection
|       |-- summarizer.py              # Extractive text summarization
|       `-- topics.py                   # Unsupervised NMF topic mining
|-- civicshield/                        # New CivicShield platform modules
|   |-- __init__.py
|   |-- config.py                       # Centralised environment-based configuration
|   |-- auth/
|   |   |-- __init__.py
|   |   |-- models.py                   # User, Role, Token Pydantic schemas
|   |   |-- jwt.py                      # JWT creation, verification, password hashing
|   |   |-- rbac.py                     # Role-based access control dependencies
|   |   `-- router.py                   # /auth/login, /auth/me, /auth/logout endpoints
|   |-- security/
|   |   |-- __init__.py
|   |   |-- schema.py                   # SecurityEvent, SecurityAlert, ThreatSummary models
|   |   |-- rules.py                    # Deterministic rule engine (brute force, escalation, etc.)
|   |   |-- anomaly.py                  # Isolation Forest anomaly detector
|   |   |-- correlator.py               # Event correlation into investigation threads
|   |   `-- generator.py               # Synthetic demo event generator (environment="demo")
|   |-- ml/
|   |   |-- __init__.py
|   |   `-- enhanced_pipeline.py        # Char n-gram features, data quality, model versioning
|   |-- shared/
|   |   |-- __init__.py
|   |   |-- logging.py                  # Structured JSON logging with structlog
|   |   `-- middleware.py               # Request-ID injection, CORS, request logging
|   |-- civic/                          # (reserved for civic module re-exports)
|   `-- streaming/                      # (reserved for SSE/WebSocket streaming)
`-- tests/
    |-- test_api.py                     # FastAPI endpoint integration tests (9)
    |-- test_auth.py                    # JWT auth, login, RBAC tests (12)
    |-- test_security.py                # Security module: rules, anomaly, generator, correlator (11)
    |-- test_ml_enhanced.py             # Enhanced ML pipeline tests (7)
    |-- test_deduplication.py           # Similarity & incident clustering tests (2)
    |-- test_explainability.py          # Token attribution tests (1)
    |-- test_model_benchmark.py         # Multi-model benchmarking & persistence tests (3)
    |-- test_pipeline.py                # End-to-end data & analytics pipeline tests (2)
    |-- test_sentiment.py               # Frustration & escalation scoring tests (3)
    `-- test_topics.py                  # Unsupervised NMF topic tests (2)
```

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🏠 Command Center | KPI cards, department pressure board, executive brief, priority queue |
| 🔍 Case Studio & XAI | Single complaint deep analysis with token attribution and frustration scoring |
| 🔗 Incident Clusters | Near-duplicate detection, deduplication metrics, incident grouping |
| 📊 Emerging Topics | Unsupervised NMF topic mining and trend discovery |
| 📂 Data Explorer | Filtered data view with search, department/urgency/ward filters |
| ⚙️ Ops Console | Operational dispatch queue with confidence-based filtering |
| 🎯 Model Benchmark | Multi-model comparison (4 algorithms, stratified K-fold CV) |
| 🛡️ Security Ops | Security event monitoring, rule-based alerts, ML anomaly detection, threat correlation |
| 🧪 ML Lab | Enhanced training with char n-grams, baseline vs enhanced comparison, model versioning |
| 📋 Data Quality | Automated quality checks, class balance analysis, dataset health monitoring |

---

## Security Module

The security module provides cybersecurity monitoring capabilities:

- **Rule Engine**: Deterministic rules for brute-force detection, privilege escalation, unusual hour activity, and high event rates.
- **Anomaly Detection**: Isolation Forest trained on numeric event features (time, category, severity, IP).
- **Event Correlation**: Groups related events by source IP within time windows into investigation threads.
- **Demo Generator**: Produces realistic synthetic security events across 7 categories (auth, network, data access, etc.) — all labelled `environment="demo"`.

---

## Authentication

The API supports JWT-based authentication with three roles:

| Role | Access Level |
|------|-------------|
| `admin` | Full access to all endpoints and settings |
| `analyst` | Access to analysis, security, and ML endpoints |
| `viewer` | Read-only access to dashboards and reports |

**Demo credentials** (development only):
- `admin` / `admin123`
- `analyst` / `analyst123`
- `viewer` / `viewer123`

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (52 tests)
python -m pytest

# Start FastAPI server
uvicorn api:app --port 8000

# Start Streamlit dashboard
streamlit run app.py --server.port 8501

# API docs
open http://localhost:8000/docs
```

---

## CI/CD

GitHub Actions workflows run automatically on push and PR to `main`:

- **test.yml**: Runs all 52 pytest tests
- **security.yml**: Bandit SAST scan + Gitleaks secret detection

---

## Tech Stack

| Area | Technology |
|------|-----------|
| ML & NLP | scikit-learn, TF-IDF, NMF, Isolation Forest |
| Backend | FastAPI, Pydantic, structlog |
| Frontend | Streamlit |
| Auth | python-jose (JWT), passlib (PBKDF2-SHA256) |
| Testing | pytest, httpx |
| Security | Bandit, Gitleaks |
| CI/CD | GitHub Actions |
| Containers | Docker, Docker Compose |

---

## Test Coverage

```
52 tests across 9 test files:
  test_api.py              —  9 tests (FastAPI endpoints)
  test_auth.py             — 12 tests (JWT, login, RBAC, middleware)
  test_security.py         — 11 tests (rules, anomaly, generator, correlator)
  test_ml_enhanced.py      —  7 tests (data quality, char features, metrics)
  test_deduplication.py    —  2 tests (similarity, clustering)
  test_explainability.py   —  1 test  (token attribution)
  test_model_benchmark.py  —  3 tests (benchmarking, persistence)
  test_pipeline.py         —  2 tests (end-to-end pipeline)
  test_sentiment.py        —  3 tests (frustration scoring)
  test_topics.py           —  2 tests (NMF topic mining)
```

---

## License

This project is for academic and educational purposes.
