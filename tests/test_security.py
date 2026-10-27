"""
Tests for CivicShield Security Module.
Covers: schema, rule engine, anomaly detector, event generator, correlator.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

import pytest

from civicshield.security.schema import (
    EventCategory,
    SecurityEvent,
    Severity,
)
from civicshield.security.rules import RuleEngine
from civicshield.security.anomaly import AnomalyDetector
from civicshield.security.generator import generate_demo_events
from civicshield.security.correlator import correlate_events


# ── Schema ───────────────────────────────────────────────────────
def test_security_event_defaults():
    event = SecurityEvent(description="Test event")
    assert event.event_id.startswith("SEC-")
    assert event.environment == "demo"
    assert event.severity == Severity.low


def test_security_event_custom():
    event = SecurityEvent(
        source_ip="10.0.0.1",
        category=EventCategory.authentication,
        severity=Severity.critical,
        description="Failed login",
        environment="demo",
    )
    assert event.source_ip == "10.0.0.1"
    assert event.category == EventCategory.authentication
    assert event.severity == Severity.critical


# ── Rule Engine ──────────────────────────────────────────────────
def test_brute_force_rule_triggers():
    engine = RuleEngine()
    now = datetime.now(timezone.utc)
    history = [
        SecurityEvent(
            source_ip="10.0.0.1",
            category=EventCategory.authentication,
            timestamp=now - timedelta(minutes=i),
        )
        for i in range(6)
    ]
    event = SecurityEvent(
        source_ip="10.0.0.1",
        category=EventCategory.authentication,
        timestamp=now,
    )
    alerts = engine.evaluate(event, history)
    rule_names = [a.rule_name for a in alerts]
    assert "brute_force_detection" in rule_names


def test_privilege_escalation_rule():
    engine = RuleEngine()
    event = SecurityEvent(
        category=EventCategory.authorization,
        severity=Severity.critical,
        description="Unauthorized admin access",
    )
    alerts = engine.evaluate(event, [])
    rule_names = [a.rule_name for a in alerts]
    assert "privilege_escalation" in rule_names


def test_no_false_positive_on_normal_event():
    engine = RuleEngine()
    event = SecurityEvent(
        category=EventCategory.system,
        severity=Severity.low,
        description="Health check passed",
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),  # noon
    )
    alerts = engine.evaluate(event, [])
    assert len(alerts) == 0


# ── Anomaly Detector ─────────────────────────────────────────────
def test_anomaly_detector_fit_and_predict():
    events = generate_demo_events(count=50, seed=99)
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(events)
    assert detector.is_fitted

    result = detector.predict(events[0])
    assert "is_anomaly" in result
    assert "anomaly_score" in result


def test_anomaly_detector_unfitted():
    detector = AnomalyDetector()
    event = SecurityEvent(description="test")
    result = detector.predict(event)
    assert result["is_anomaly"] is False


# ── Generator ────────────────────────────────────────────────────
def test_generate_demo_events_count():
    events = generate_demo_events(count=25)
    assert len(events) == 25
    assert all(e.environment == "demo" for e in events)


def test_generate_demo_events_sorted():
    events = generate_demo_events(count=20)
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps, reverse=True)


# ── Correlator ───────────────────────────────────────────────────
def test_correlate_groups_by_ip():
    now = datetime.now(timezone.utc)
    events = [
        SecurityEvent(source_ip="10.0.0.1", timestamp=now - timedelta(minutes=i))
        for i in range(5)
    ]
    threads = correlate_events(events, time_window_minutes=30)
    assert len(threads) >= 1
    assert threads[0]["source_ip"] == "10.0.0.1"
    assert threads[0]["event_count"] == 5


def test_correlate_separates_distant_events():
    now = datetime.now(timezone.utc)
    events = [
        SecurityEvent(source_ip="10.0.0.1", timestamp=now),
        SecurityEvent(source_ip="10.0.0.1", timestamp=now - timedelta(hours=2)),
    ]
    threads = correlate_events(events, time_window_minutes=15)
    # Two events are too far apart, each cluster has only 1 event — below threshold
    assert len(threads) == 0
