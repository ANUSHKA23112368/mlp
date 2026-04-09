from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import build_overview
from civiclens.data import load_demo_data
from civiclens.modeling import train_models
from civiclens.summarizer import summarize_text


def test_training_and_prediction_pipeline():
    frame = load_demo_data()
    artifacts = train_models(frame)
    predictions = artifacts.predict(
        [
            "Garbage has not been collected from our lane for three days and the smell is getting worse.",
            "The hospital billing desk charged my father twice and nobody is resolving the refund.",
        ]
    )

    assert list(predictions.columns) == [
        "complaint_text",
        "predicted_department",
        "predicted_urgency",
    ]
    assert len(predictions) == 2


def test_analytics_and_summary():
    frame = load_demo_data()
    artifacts = train_models(frame)
    predictions = artifacts.predict(frame["complaint_text"].tolist())
    overview = build_overview(predictions)

    assert overview["total_cases"] == len(frame)
    assert isinstance(overview["keywords"], list)
    assert overview["average_risk_score"] > 0

    summary = summarize_text(
        "Water has been leaking for days. Multiple homes are affected. Officials have not responded."
    )
    assert len(summary) > 20

