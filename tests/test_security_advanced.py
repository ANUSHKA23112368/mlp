from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from civicshield.security import generate_demo_events, RuleEngine, AnomalyDetector, correlate_events


def test_security_anomaly_detector_training():
    events = generate_demo_events(count=40, hours_back=12, seed=99)
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(events)
    assert detector.model is not None

    alerts = [a for e in events if (a := detector.detect(e)) is not None]
    assert isinstance(alerts, list)


def test_security_event_temporal_correlation():
    events = generate_demo_events(count=30, hours_back=6, seed=123)
    threads = correlate_events(events, time_window_minutes=20)
    assert isinstance(threads, list)
    for t in threads:
        assert "thread_id" in t
        assert "source_ip" in t
        assert "event_count" in t
        assert t["event_count"] >= 1
