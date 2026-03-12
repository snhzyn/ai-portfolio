"""
Simple deduplication helpers.

This module provides a lightweight event clustering approach for the MVP.
It groups highly similar events and keeps the strongest representative event
based on confidence.
"""

from __future__ import annotations

from collections import defaultdict

from app.schemas.event import EventItem


def deduplicate_events(events: list[EventItem]) -> tuple[list[EventItem], dict[str, list[str]]]:
    """
    Deduplicate events using a simple clustering key.

    Current clustering key:
    - worker
    - event_type
    - market_impact

    Returns:
    - deduplicated event list
    - mapping from kept event headline to supporting source names
    """
    
    clusters: dict[tuple[str, str, str], list[EventItem]] = defaultdict(list)

    for event in events:
        key = (
            event.worker.value,
            event.event_type,
            event.market_impact,
        )
        clusters[key].append(event)

    deduplicated: list[EventItem] = []
    supporting_sources_map: dict[str, list[str]] = {}

    for cluster_events in clusters.values():
        best_event = max(cluster_events, key=lambda e: e.confidence)
        deduplicated.append(best_event)

        supporting_sources_map[best_event.headline] = [
            e.source for e in cluster_events if e.source != best_event.source
        ]

    deduplicated.sort(key=lambda e: (e.importance, e.confidence), reverse=True)
    return deduplicated, supporting_sources_map