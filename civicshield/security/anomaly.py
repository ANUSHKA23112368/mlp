"""
CivicShield Security — Anomaly Detection
Uses Isolation Forest on numeric features extracted from events.
"""
from __future__ import annotations
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from civicshield.security.schema import (
    EventCategory,
    SecurityAlert,
    SecurityEvent,
    Severity,
)


# Map categories to integers for feature extraction
_CATEGORY_MAP = {c: i for i, c in enumerate(EventCategory)}
_SEVERITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _event_to_features(event: SecurityEvent) -> list[float]:
    """Extract numeric feature vector from a SecurityEvent."""
    return [
        event.timestamp.hour,
        event.timestamp.minute,
        event.timestamp.weekday(),
        _CATEGORY_MAP.get(event.category, 0),
        _SEVERITY_MAP.get(event.severity.value, 0),
        len(event.description),
        sum(int(octet) for octet in event.source_ip.split(".") if octet.isdigit()),
    ]


class AnomalyDetector:
    """Isolation Forest wrapper for security event anomaly detection."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.is_fitted = False

    def fit(self, events: list[SecurityEvent]) -> None:
        if len(events) < 10:
            return
        X = np.array([_event_to_features(e) for e in events])
        self.model.fit(X)
        self.is_fitted = True

    def predict(self, event: SecurityEvent) -> dict[str, Any]:
        """Return anomaly score and label for a single event."""
        if not self.is_fitted:
            return {"is_anomaly": False, "anomaly_score": 0.0}
        X = np.array([_event_to_features(event)])
        label = self.model.predict(X)[0]       # -1 = anomaly, 1 = normal
        score = -self.model.score_samples(X)[0]  # higher = more anomalous
        return {"is_anomaly": label == -1, "anomaly_score": round(float(score), 4)}

    def detect(self, event: SecurityEvent) -> SecurityAlert | None:
        """Run anomaly detection; return alert if anomalous."""
        result = self.predict(event)
        if not result["is_anomaly"]:
            return None
        return SecurityAlert(
            title="ML Anomaly Detected",
            description=(
                f"Isolation Forest flagged event {event.event_id} as anomalous "
                f"(score: {result['anomaly_score']:.3f}). "
                f"Category: {event.category.value}, Source: {event.source_ip}"
            ),
            severity=Severity.medium,
            category=event.category,
            source_events=[event.event_id],
            rule_name="isolation_forest_anomaly",
            confidence=min(result["anomaly_score"] / 1.0, 0.99),
            environment=event.environment,
        )
