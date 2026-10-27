"""
CivicShield Security — Demo Event Generator
Generates realistic synthetic security events for dashboard demos.
ALL events are labelled environment="demo".
"""
from __future__ import annotations
import random
from datetime import datetime, timedelta, timezone

from civicshield.security.schema import (
    EventCategory,
    SecurityEvent,
    Severity,
)

_SERVICES = [
    "api-gateway", "auth-service", "civic-classifier", "dashboard",
    "postgres-db", "redis-cache", "file-storage", "webhook-handler",
]

_IPS = [
    "192.168.1.100", "10.0.0.42", "172.16.0.8", "203.0.113.77",
    "198.51.100.12", "192.168.2.55", "10.10.10.1", "172.31.0.99",
]

_DESCRIPTIONS = {
    EventCategory.authentication: [
        "Failed login attempt for user 'operator'",
        "Password reset requested from unknown device",
        "Token expired during active session",
        "Multiple MFA failures detected",
        "Login from new geographic location",
    ],
    EventCategory.authorization: [
        "User attempted to access admin panel without permission",
        "API key used outside allowed IP range",
        "Role escalation request denied",
        "Unauthorized export of complaint records attempted",
    ],
    EventCategory.network: [
        "Unusual outbound traffic spike detected",
        "Port scan detected from external IP",
        "DNS query to suspicious domain blocked",
        "SSL certificate mismatch on internal service",
    ],
    EventCategory.data_access: [
        "Bulk data export triggered by non-admin user",
        "Database query returned unusually large result set",
        "Sensitive field accessed without audit flag",
    ],
    EventCategory.system: [
        "Service restart triggered automatically",
        "CPU usage exceeded 95% threshold",
        "Disk space below 10% on worker node",
        "Container health check failed 3 consecutive times",
    ],
    EventCategory.anomaly: [
        "Unusual API call pattern detected",
        "Request rate 5x above baseline",
        "Unexpected payload structure in webhook",
    ],
    EventCategory.policy_violation: [
        "Data retention policy breach: records older than 90 days accessed",
        "Unencrypted data transfer attempt blocked",
        "API response included PII in debug mode",
    ],
}


def generate_demo_events(
    count: int = 50,
    hours_back: int = 24,
    seed: int = 42,
) -> list[SecurityEvent]:
    """Generate `count` realistic synthetic security events."""
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    events: list[SecurityEvent] = []

    for _ in range(count):
        category = rng.choice(list(EventCategory))
        severity_weights = [0.35, 0.35, 0.20, 0.10]
        severity = rng.choices(list(Severity), weights=severity_weights, k=1)[0]
        ts = now - timedelta(
            hours=rng.uniform(0, hours_back),
            minutes=rng.uniform(0, 60),
        )
        descriptions = _DESCRIPTIONS.get(category, ["Unknown event"])
        events.append(
            SecurityEvent(
                timestamp=ts,
                source_ip=rng.choice(_IPS),
                source_service=rng.choice(_SERVICES),
                category=category,
                description=rng.choice(descriptions),
                severity=severity,
                environment="demo",
            )
        )

    events.sort(key=lambda e: e.timestamp, reverse=True)
    return events
