# CivicLens AI

CivicLens AI is a resume-ready ML and NLP project for analyzing customer tickets, public grievance logs, or service complaints. It helps operations teams identify urgent issues faster, route cases to the right department, and summarize the biggest pain points from unstructured text.

## Why this project is strong for a resume

- Solves a real operational problem with measurable business and civic impact
- Uses end-to-end NLP and machine learning instead of only calling an LLM API
- Includes model training, evaluation, analytics, a deployable web app, and API support
- Can be adapted to customer support, municipal complaints, healthcare feedback, or internal service desks

## Core features

- Department classification from complaint text
- Urgency prediction for triaging high-risk complaints
- Complaint trend analytics and keyword intelligence
- Extractive issue summaries for leadership-friendly reporting
- Streamlit dashboard for interactive analysis
- Dataset explorer for label and text distribution analysis
- Operations view for filtering complaint queues by department and urgency
- Batch complaint analysis table for uploaded CSV files
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

## Quick start

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Recommended local/runtime Python version: `3.10` to `3.12`.

## Use in VS Code

1. Open the project folder in VS Code.
2. Run the task `Create venv (Python 3.10)` if `.venv` does not exist yet.
3. Run the task `Install dependencies`.
4. Press `F5` and choose `Streamlit App` or `FastAPI API`.
5. Run `Pytest` from the Run and Debug panel or the `Run tests` task.

The repo already includes `.vscode/settings.json`, `launch.json`, and `tasks.json` so you can start directly in VS Code without extra setup.

## Run the API

```bash
uvicorn api:app --reload
```

## Deploy

### Streamlit Community Cloud

1. Push the repo to GitHub.
2. Create a new Streamlit app from the repository.
3. Set the app entry point to `app.py`.
4. Let it install from `requirements.txt`.

Recommended files for this route:
- `requirements.txt`
- `runtime.txt`

### Render or Railway

For Render, this repo now includes `render.yaml`.

1. Push the repo to GitHub.
2. In Render, create a new Blueprint or Web Service from the repository.
3. Render will use:
   - build command: `pip install -r requirements.txt`
   - start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

For Docker-based deployment, the repo also includes:
- `Dockerfile`
- `.dockerignore`

## Suggested resume bullet

Built an end-to-end NLP complaint intelligence platform using Python, scikit-learn, FastAPI, and Streamlit to classify issue ownership and urgency, generate executive summaries, and surface service risk trends from unstructured complaint data.
