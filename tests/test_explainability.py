from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.data import load_demo_data
from civiclens.explainability import explain_prediction
from civiclens.modeling import train_models


def test_explain_prediction():
    df = load_demo_data()
    artifacts = train_models(df)
    
    # Test department explanation
    res = explain_prediction(
        artifacts.department_model,
        "The drinking water pipeline ruptured and caused flooding in our neighborhood.",
        top_k=5,
    )
    
    assert res["target_class"] == "water"
    assert res["confidence"] > 0
    assert len(res["positive_features"]) > 0
    assert any("water" in item["token"] or "pipe" in item["token"] for item in res["positive_features"])
    assert "<mark" in res["highlighted_html"]

    # Test urgency explanation
    urg_res = explain_prediction(
        artifacts.urgency_model,
        "Dangerous open manhole near school caused an accident immediately!",
        top_k=5,
    )
    assert urg_res["target_class"] in ["critical", "high", "medium", "low"]
    assert isinstance(urg_res["top_features"], list)
