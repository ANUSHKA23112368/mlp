from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import build_geo_and_channel_overview, enrich_predictions
from civiclens.data import load_demo_data
from civiclens.modeling import train_models


def test_geo_and_channel_overview_generation():
    data = load_demo_data()
    artifacts = train_models(data)
    preds = enrich_predictions(artifacts.predict(data["complaint_text"].tolist()))
    for col in ["ward", "channel", "timestamp", "status"]:
        if col in data.columns:
            preds[col] = data[col]
            
    geo_overview = build_geo_and_channel_overview(preds)
    assert "ward_distribution" in geo_overview
    assert "channel_distribution" in geo_overview
    assert len(geo_overview["ward_distribution"]) > 0
    assert len(geo_overview["channel_distribution"]) > 0
