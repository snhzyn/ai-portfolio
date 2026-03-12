"""
Source fetching utilities for FINSIGHT.

Step 12 improves source parsing quality by using BeautifulSoup for cleaner
title and snippet extraction while preserving fallback behavior.
"""

from __future__ import annotations

import html
from datetime import date
from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.config.source_registry import SOURCE_REGISTRY
from app.services.source_models import RawSourceItem
from app.services.sources.us.fed import fetch_fed_press_release_item


def fetch_worker_sources(worker_name: str, target_date: date) -> list[RawSourceItem]:
    """
    Fetch raw source items for a given worker.

    Strategy:
    - Try real HTTP fetch for selected supported sources
    - Fall back to deterministic placeholder items if fetch/parsing fails
    """
    worker_config = SOURCE_REGISTRY.get(worker_name, {})
    primary_sources = worker_config.get("primary", [])
    max_items = worker_config.get("max_items", 5)

    raw_items: list[RawSourceItem] = []

    for source in primary_sources[:max_items]:
        fetched_item = _try_fetch_supported_source(source=source, target_date=target_date)
        if fetched_item is not None:
            raw_items.append(fetched_item)
        else:
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


def _try_fetch_supported_source(source: dict[str, Any], target_date: date) -> RawSourceItem | None:
    source_name = source["name"]

    print(f"[FETCH] source={source_name}")

    try:
        if source_name == "Federal Reserve FOMC":
            item = fetch_fed_press_release_item(source=source, target_date=target_date)
            print(f"[FETCH][FED] item is None = {item is None}")
            return item

        if source_name == "U.S. BLS CPI Releases":
            print("[FETCH][BLS] handler not implemented yet")
            return None

    except Exception as exc:
        print(f"[FETCH] fetch failed for {source_name}: {exc}")
        return None

    print(f"[FETCH] unsupported real-fetch source: {source_name}")
    return None


def _build_dummy_headline(worker_name: str, source_name: str) -> str:
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


def _build_dummy_content(worker_name: str, source_name: str, category: str) -> str:
    """
    Generate deterministic dummy content by worker type and source category.
    """
    return (
        f"This is a placeholder raw item for worker '{worker_name}' from source "
        f"'{source_name}' under category '{category}'. "
        "It will later be replaced by real fetched source content."
    )



