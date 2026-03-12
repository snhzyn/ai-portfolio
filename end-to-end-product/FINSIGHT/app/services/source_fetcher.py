"""
Source fetching utilities for FINSIGHT.

A retrieval abstraction layer so workers no longer hardcode EventItem objects directly.

This module returns deterministic dummy source items based on
the configured source registry. 

Later, real HTTP fetching/parsing logic
will replace the placeholder implementations.
"""

from __future__ import annotations

from datetime import date

from app.config.source_registry import SOURCE_REGISTRY
from app.services.source_models import RawSourceItem


def fetch_worker_sources(worker_name: str, target_date: date) -> list[RawSourceItem]:
    """
    Fetch raw source items for a given worker.

    A deterministic placeholder implementation that converts configured registry entries into mock raw items.

    Later versions should:
    - request actual web pages / APIs
    - parse source content
    - filter by publish date
    """

    worker_config = SOURCE_REGISTRY.get(worker_name, {})
    primary_sources = worker_config.get("primary", [])
    max_items = worker_config.get("max_items", 5)

    raw_items: list[RawSourceItem] = []

    for source in primary_sources[:max_items]:
        raw_items.append(
            RawSourceItem(
                source_name=source["name"],
                source_type=source["source_type"],
                category=source["category"],
                url=source["base_url"] or "https://example.com",
                headline=_build_dummy_headline(worker_name, source["name"]),
                publish_date=target_date,
                content=_build_dummy_content(worker_name, source["name"], source["category"]),
            )
        )

    return raw_items


def _build_dummy_headline(worker_name: str, source_name: str):
    """
    Generate deterministic dummy headlines by worker type.
    """

    if worker_name == "macro":
        return f"{source_name}: Policy and macro update affects global rate expectations"
    if worker_name == "markets":
        return f"{source_name}: Equity markets react to shifting risk sentiment"
    if worker_name == "commodities_fx":
        return f"{source_name}: Oil, FX, and yield moves reshape market positioning"
    if worker_name == "geopolitical":
        return f"{source_name}: New geopolitical developments raise cross-market concerns"
    return f"{source_name}: Market-relevant update"


def _build_dummy_content(worker_name: str, source_name: str, category: str):
    """
    Generate deterministic dummy content by worker type and source category.
    """

    return (
        f"This is a placeholder raw item for worker '{worker_name}' from source "
        f"'{source_name}' under category '{category}'. "
        "It will later be replaced by real fetched source content."
    )