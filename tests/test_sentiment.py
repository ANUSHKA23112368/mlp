from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.sentiment import analyze_sentiment, batch_analyze_sentiment


def test_sentiment_critical_escalation():
    text = "EMERGENCY: The electric transformer exploded and fire broke out! This is the third time nobody took action!"
    res = analyze_sentiment(text)
    assert res["frustration_score"] >= 65
    assert res["escalation_risk"] in ["High Risk", "Critical Flashpoint"]
    assert res["sentiment_score"] < 0
    assert "fire" in res["frustration_signals"] or "explod" in res["frustration_signals"]
    assert "dedicated" in res["liaison_guidance"].lower() or "senior" in res["liaison_guidance"].lower()


def test_sentiment_normal_routine():
    text = "The bus display board shows the wrong terminal destination for route 42."
    res = analyze_sentiment(text)
    assert res["frustration_score"] < 40
    assert res["escalation_risk"] in ["Normal", "Elevated"]
    assert "routine" in res["liaison_guidance"].lower() or "standard" in res["liaison_guidance"].lower()


def test_batch_analyze_sentiment():
    texts = [
        "Dangerous open manhole near the school entrance!",
        "Minor billing question regarding electricity meter reading.",
    ]
    results = batch_analyze_sentiment(texts)
    assert len(results) == 2
    assert results[0]["frustration_score"] > results[1]["frustration_score"]
