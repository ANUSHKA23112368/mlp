"""
CivicShield Security — Rule Engine
Deterministic rule-based detection on SecurityEvents.
Rules run before ML anomaly detection and produce immediate alerts.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

from civicshield.security.schema import (
    EventCategory,
    SecurityAlert,
    SecurityEvent,
    Severity,
)


class RuleEngine:
    """Stateless rule evaluator.

    Each rule is a callable ``(event, history) -> Optional[SecurityAlert]``.
    """

    def __init__(self):
        self.rules = [
            self._brute_force_rule,
            self._privilege_escalation_rule,
            self._unusual_hour_rule,
            self._high_rate_rule,
        ]

    def evaluate(
        self, event: SecurityEvent, history: list[SecurityEvent]
    ) -> list[SecurityAlert]:
        alerts: list[SecurityAlert] = []
        for rule_fn in self.rules:
            alert = rule_fn(event, history)
            if alert is not None:
                alerts.append(alert)
        return alerts

    # ── Individual Rules ─────────────────────────────────────────

    def _brute_force_rule(
        self, event: SecurityEvent, history: list[SecurityEvent]
    ) -> Optional[SecurityAlert]:
        """Flag >5 failed auth attempts from same IP in 10 min window."""
        if event.category != EventCategory.authentication:
            return None
        window = event.timestamp - timedelta(minutes=10)
        same_ip = [
            e for e in history
            if e.source_ip == event.source_ip
            and e.category == EventCategory.authentication
            and e.timestamp >= window
        ]
        if len(same_ip) < 5:
            return None
        return SecurityAlert(
            title="Brute-Force Login Attempt Detected",
            description=f"{len(same_ip)} failed auth attempts from {event.source_ip} in 10 min.",
            severity=Severity.high,
            category=EventCategory.authentication,
            source_events=[e.event_id for e in same_ip],
            rule_name="brute_force_detection",
            confidence=0.88,
            environment=event.environment,
        )

    def _privilege_escalation_rule(
        self, event: SecurityEvent, history: list[SecurityEvent]
    ) -> Optional[SecurityAlert]:
        """Flag any authorization violation event as potential escalation."""
        if event.category != EventCategory.authorization:
            return None
        if event.severity not in (Severity.high, Severity.critical):
            return None
        return SecurityAlert(
            title="Privilege Escalation Attempt",
            description=f"High-severity authorization event from {event.source_service}: {event.description}",
            severity=Severity.critical,
            category=EventCategory.authorization,
            source_events=[event.event_id],
            rule_name="privilege_escalation",
            confidence=0.75,
            environment=event.environment,
        )

    def _unusual_hour_rule(
        self, event: SecurityEvent, history: list[SecurityEvent]
    ) -> Optional[SecurityAlert]:
        """Flag activity outside 06:00-22:00 UTC as suspicious."""
        hour = event.timestamp.hour
        if 6 <= hour <= 22:
            return None
        return SecurityAlert(
            title="Unusual Hour Activity",
            description=f"Activity from {event.source_ip} at {event.timestamp.strftime('%H:%M')} UTC.",
            severity=Severity.medium,
            category=event.category,
            source_events=[event.event_id],
            rule_name="unusual_hour",
            confidence=0.55,
            environment=event.environment,
        )

    def _high_rate_rule(
        self, event: SecurityEvent, history: list[SecurityEvent]
    ) -> Optional[SecurityAlert]:
        """Flag >20 events from same source in 5 minutes."""
        window = event.timestamp - timedelta(minutes=5)
        same_source = [
            e for e in history
            if e.source_ip == event.source_ip and e.timestamp >= window
        ]
        if len(same_source) < 20:
            return None
        return SecurityAlert(
            title="High Event Rate Detected",
            description=f"{len(same_source)} events from {event.source_ip} in 5 min — possible DoS or scanner.",
            severity=Severity.high,
            category=EventCategory.network,
            source_events=[e.event_id for e in same_source[-20:]],
            rule_name="high_rate",
            confidence=0.72,
            environment=event.environment,
        )
