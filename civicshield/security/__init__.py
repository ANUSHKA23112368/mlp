"""
CivicShield Security — Package __init__
"""
from civicshield.security.schema import (
    EventCategory,
    SecurityAlert,
    SecurityEvent,
    Severity,
    ThreatSummary,
)
from civicshield.security.rules import RuleEngine
from civicshield.security.anomaly import AnomalyDetector
from civicshield.security.generator import generate_demo_events
from civicshield.security.correlator import correlate_events

__all__ = [
    "EventCategory",
    "SecurityAlert",
    "SecurityEvent",
    "Severity",
    "ThreatSummary",
    "RuleEngine",
    "AnomalyDetector",
    "generate_demo_events",
    "correlate_events",
]
