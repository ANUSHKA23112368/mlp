from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import build_executive_report, enrich_predictions
from civiclens.data import load_demo_data
from civiclens.modeling import train_models


def test_executive_report_generation():
    data = load_demo_data()
    artifacts = train_models(data)
    preds = enrich_predictions(artifacts.predict(data["complaint_text"].tolist()))
    for col in ["ward", "channel", "timestamp", "status"]:
        if col in data.columns:
            preds[col] = data[col]
            
    report = build_executive_report(preds)
    assert isinstance(report, str)
    assert len(report) > 50
    assert "CivicLens" in report or "Executive Brief" in report or "Intelligence" in report
