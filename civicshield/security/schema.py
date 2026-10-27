"""
CivicShield Security — Event Schema
Pydantic models for security events, alerts, and threat classification.
All synthetic/demo events carry environment="demo" by design.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class EventCategory(str, Enum):
    authentication = "authentication"
    authorization = "authorization"
    network = "network"
    data_access = "data_access"
    system = "system"
    anomaly = "anomaly"
    policy_violation = "policy_violation"


class SecurityEvent(BaseModel):
    """Inbound security event from any source (SIEM, app logs, simulated)."""
    event_id: str = Field(default_factory=lambda: f"SEC-{uuid.uuid4().hex[:8].upper()}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_ip: str = "0.0.0.0"
    source_service: str = "unknown"
    category: EventCategory = EventCategory.system
    description: str = ""
    severity: Severity = Severity.low
    raw_payload: dict = Field(default_factory=dict)
    environment: str = "demo"          # ALWAYS "demo" for synthetic data
    is_resolved: bool = False
    assigned_to: Optional[str] = None


class SecurityAlert(BaseModel):
    """Processed alert derived from one or more SecurityEvents."""
    alert_id: str = Field(default_factory=lambda: f"ALT-{uuid.uuid4().hex[:8].upper()}")
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str
    description: str
    severity: Severity
    category: EventCategory
    source_events: list[str] = Field(default_factory=list)
    rule_name: str = ""
    confidence: float = 0.0
    is_false_positive: bool = False
    environment: str = "demo"


class ThreatSummary(BaseModel):
    """Roll-up analytics for the security dashboard."""
    total_events: int = 0
    total_alerts: int = 0
    critical_alerts: int = 0
    high_alerts: int = 0
    medium_alerts: int = 0
    low_alerts: int = 0
    top_categories: list[dict] = Field(default_factory=list)
    top_source_ips: list[dict] = Field(default_factory=list)
    recent_events: list[SecurityEvent] = Field(default_factory=list)
    environment: str = "demo"
