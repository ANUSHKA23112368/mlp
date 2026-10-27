"""
CivicShield Security — Event Correlator
Groups related security events into investigation threads.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import timedelta

from civicshield.security.schema import SecurityEvent, SecurityAlert, Severity


def correlate_events(
    events: list[SecurityEvent],
    time_window_minutes: int = 15,
) -> list[dict]:
    """Group events by source_ip within a time window into correlated threads."""
    if not events:
        return []

    # Sort by timestamp
    sorted_events = sorted(events, key=lambda e: e.timestamp)

    # Group by source IP
    ip_groups: dict[str, list[SecurityEvent]] = defaultdict(list)
    for event in sorted_events:
        ip_groups[event.source_ip].append(event)

    threads: list[dict] = []
    thread_id = 0

    for ip, ip_events in ip_groups.items():
        # Split into time-windowed clusters
        current_cluster: list[SecurityEvent] = [ip_events[0]]

        for event in ip_events[1:]:
            if (event.timestamp - current_cluster[-1].timestamp) <= timedelta(minutes=time_window_minutes):
                current_cluster.append(event)
            else:
                if len(current_cluster) >= 2:
                    thread_id += 1
                    threads.append(_build_thread(thread_id, ip, current_cluster))
                current_cluster = [event]

        if len(current_cluster) >= 2:
            thread_id += 1
            threads.append(_build_thread(thread_id, ip, current_cluster))

    threads.sort(key=lambda t: t["max_severity_level"], reverse=True)
    return threads


_SEV_LEVEL = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _build_thread(thread_id: int, ip: str, events: list[SecurityEvent]) -> dict:
    categories = list({e.category.value for e in events})
    max_sev = max(events, key=lambda e: _SEV_LEVEL.get(e.severity.value, 0))
    return {
        "thread_id": thread_id,
        "source_ip": ip,
        "event_count": len(events),
        "categories": categories,
        "max_severity": max_sev.severity.value,
        "max_severity_level": _SEV_LEVEL.get(max_sev.severity.value, 0),
        "time_span_minutes": round(
            (events[-1].timestamp - events[0].timestamp).total_seconds() / 60, 1
        ),
        "first_seen": events[0].timestamp.isoformat(),
        "last_seen": events[-1].timestamp.isoformat(),
        "event_ids": [e.event_id for e in events],
        "descriptions": [e.description for e in events[:5]],
    }
