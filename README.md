# CivicLens AI

CivicLens AI is a ML and NLP project for analyzing customer tickets, public grievance logs, or service complaints. It helps operations teams identify urgent issues faster, route cases to the right department, and summarize the biggest pain points from unstructured text.

## HIGHS
- Solves a real operational problem with measurable business and civic impact
- Uses end-to-end NLP and machine learning instead of only calling an LLM API
- Includes model training, evaluation, analytics, a deployable web app, and API support
- Can be adapted to customer support, municipal complaints, healthcare feedback, or internal service desks

## Core features

- Department classification from complaint text
- Urgency prediction for triaging high-risk complaints
- Confidence scoring for routing and urgency predictions
- Complaint trend analytics and keyword intelligence
- Extractive issue summaries for leadership-friendly reporting
- Command-center style Streamlit dashboard with a more polished UI
- Case studio for single complaint analysis with SLA and action guidance
- Dataset explorer for label, text, and prediction comparison analysis
- Operations console for filtering complaint queues by department, urgency, and confidence
- Batch complaint analysis table plus CSV and brief export tools
- FastAPI service for deployment or portfolio demos

## Project structure

```text
.
|-- app.py
|-- api.py
|-- data/
|   `-- demo_complaints.csv
|-- src/
|   `-- civiclens/
|       |-- analytics.py
|       |-- data.py
|       |-- modeling.py
|       `-- summarizer.py
|-- tests/
|   `-- test_pipeline.py
|-- Dockerfile
`-- requirements.txt
```
