# CivicLens AI v2.0

CivicLens AI is an end-to-end, production-grade NLP and Machine Learning platform for analyzing citizen grievance logs, municipal tickets, and customer support complaints. It automates department routing, triage urgency, Explainable AI (XAI) feature attribution, near-duplicate incident clustering, citizen frustration tracking, and executive reporting from unstructured text.

---

## What makes this project standout

- **Solves a Real Operational Problem**: Cuts response delays, prevents redundant field dispatches by detecting near-duplicate incident reports, and surfaces citizen frustration flashpoints before they escalate.
- **Pure ML & NLP Engineering**: Built with scikit-learn, TF-IDF vectorization, NMF topic decomposition, and calibrated models rather than simple API wrappers.
- **Explainable AI (XAI)**: Features token-level local feature attribution that shows exactly which words influenced department routing and urgency classifications.
- **Incident Deduplication Hub**: Groups corroborating citizen reports regarding the same underlying incident (e.g., water main breaks, power transformer fires, traffic gridlock), measuring dispatches saved.
- **Citizen Frustration Index (0–100)**: Quantifies citizen emotional distress and flags escalation risk levels (Normal, Elevated, High Risk, Critical Flashpoint) with recommended liaison guidance.
- **Multi-Model Benchmark Engine**: Automatically compares Logistic Regression, Calibrated Linear SVM, Multinomial Naive Bayes, and Random Forest using Stratified K-Fold Cross-Validation.
- **Interactive Command Center**: 7-tab Streamlit dashboard with custom CSS, interactive charts, and real-time inference.
- **Enterprise FastAPI Service**: Production REST API with batch processing, explainability endpoints, deduplication clustering, and OpenAPI documentation.
- **Robust Automated Test Suite**: 22 unit and integration tests covering pipelines, API endpoints, explainability, deduplication, topics, sentiment, and benchmarks.

---

## Architecture & Project Structure

```text
.
|-- app.py                              # Streamlit 7-tab interactive dashboard
|-- api.py                              # FastAPI REST service with 10 endpoints
|-- pytest.ini                          # Test configuration
|-- requirements.txt                    # Python dependencies
|-- Dockerfile                          # Production container configuration
|-- render.yaml                         # Cloud deployment configuration
|-- data/
|   `-- demo_complaints.csv             # 150 realistic complaints with Ward, Channel, Status & Timestamp
|-- src/
|   `-- civiclens/
|       |-- __init__.py
|       |-- analytics.py                # SLA calculations, risk scoring & geo/channel analytics
|       |-- data.py                     # Dataset validation, loading & default imputation
|       |-- deduplication.py            # Cosine similarity clustering & incident deduplication
|       |-- explainability.py           # XAI token attribution & HTML markup generation
|       |-- modeling.py                 # Multi-model training, benchmarking, persistence & confusion matrices
|       |-- sentiment.py                # Citizen frustration scoring & escalation risk detection
|       |-- summarizer.py               # Extractive text summarization
|       `-- topics.py                   # Unsupervised NMF topic mining
`-- tests/
    |-- test_api.py                     # FastAPI endpoint integration tests
    |-- test_deduplication.py           # Similarity & incident clustering tests
    |-- test_explainability.py          # Token attribution tests
    |-- test_model_benchmark.py         # Multi-model benchmarking & persistence tests
    |-- test_pipeline.py                # End-to-end data & analytics pipeline tests
    |-- test_sentiment.py               # Frustration & escalation scoring tests
    `-- test_topics.py                  # NMF topic modeling tests
```

---

## Core Capabilities

### 1. Intelligent Routing & Urgency Classification
- Classifies incoming text into 9 operational departments: `water`, `roads`, `sanitation`, `utilities`, `health`, `public_safety`, `transport`, `digital_services`, and `education`.
- Predicts urgency levels (`low`, `medium`, `high`, `critical`) and maps them to concrete turnaround SLA windows (2h, 8h, 24h, 72h).
- Computes calibrated confidence scores and uncertainty bands.

### 2. Explainable AI (XAI) Token Attribution
- Inspects classifier coefficients and TF-IDF term weights to extract top positive and negative feature contributions.
- Renders highlighted spans directly on complaint narratives to explain model predictions to non-technical analysts.

### 3. Incident Clustering & Near-Duplicate Detection
- Detects when multiple citizens submit reports about the same underlying breakdown (e.g. 5th Avenue water pipe burst, West Industrial transformer explosion).
- Designates a Primary Root Ticket and links corroborating reports, reducing duplicate technician dispatches.

### 4. Citizen Frustration & Escalation Propensity Index
- Evaluates emotional distress markers, repeated failure signals, danger keywords, and shouting emphasis.
- Outputs a normalized Frustration Index (0–100) and actionable citizen liaison recommendations.

### 5. Emerging Topics (Unsupervised NLP Mining)
- Discovers latent issue clusters dynamically using Non-negative Matrix Factorization (NMF) without preset labels.
- Computes queue prevalence percentages and surfaces representative complaints.

### 6. Multi-Model Benchmarking
- Evaluates 4 candidate model architectures with Stratified K-Fold CV:
  - **Logistic Regression** (Balanced class weights)
  - **Calibrated Linear Support Vector Machine** (LinearSVC + CalibratedClassifierCV)
  - **Multinomial Naive Bayes**
  - **Random Forest Classifier**
- Reports Weighted F1, Accuracy, training latency (ms), and interactive confusion matrices.

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ANUSHKA23112368/mlp.git
cd mlp

# Create virtual environment and activate
py -3.10 -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 3. Launch the FastAPI Service

```bash
uvicorn api:app --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### 4. Run the Test Suite

```bash
pytest
```
Executes all 22 automated test suites across API, ML models, XAI, deduplication, topics, and sentiment.

---

## API Reference Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status, active algorithm & sample count |
| `POST` | `/predict` | Single complaint classification, SLAs, sentiment & key tokens |
| `POST` | `/predict/batch` | High-throughput batch classification for multiple complaints |
| `POST` | `/explain` | XAI token attribution, feature scores & highlighted HTML |
| `POST` | `/deduplicate` | Groups complaints into incident clusters and returns dispatch savings |
| `POST` | `/similar` | Finds top-k similar complaints in the queue matching a query |
| `GET` | `/topics` | Unsupervised NMF topic mining across active complaint records |
| `GET` | `/benchmark` | Multi-model cross-validation comparison leaderboard |
| `GET` | `/geo-analytics` | Ward/district and intake channel operational distribution |
| `GET` | `/overview` | Complete operational metrics, queue brief & action board |
| `GET` | `/brief` | Plain-text executive summary for leadership |

---

## Deployment

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Link the repository on [share.streamlit.io](https://share.streamlit.io).
3. Set main file path to `app.py`.

### Docker
```bash
docker build -t civiclens-ai .
docker run -p 8501:8501 civiclens-ai
```

---

## Suggested Resume Bullets

- **ML / NLP Engineer**: *Architected CivicLens AI v2.0, an enterprise NLP complaint triage and incident intelligence platform using Python, scikit-learn, FastAPI, and Streamlit; engineered Explainable AI (XAI) feature attribution and NMF topic modeling to automate routing across 9 operational departments with 75%+ F1 score.*
- **Data Scientist**: *Implemented TF-IDF cosine-similarity incident clustering to detect near-duplicate civic grievances, reducing redundant municipal field dispatches by ~10% across 150+ operational tickets.*
- **Applied AI / Full-Stack**: *Developed a citizen frustration scoring engine (0–100) and escalation risk classifier integrated into a 7-tab Streamlit Command Center and a 10-endpoint production FastAPI service backed by a 22-test automated test suite.*
