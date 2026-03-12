"""
Source fetching utilities for FINSIGHT.

Step 11 introduces real HTTP fetching for selected sources while preserving
a safe placeholder fallback for unsupported or failed sources.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import httpx

from app.config.source_registry import SOURCE_REGISTRY
from app.services.source_models import RawSourceItem


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
    """
    Try to fetch and minimally parse supported official sources.

    Currently supported:
    - Federal Reserve FOMC
    - U.S. BLS CPI Releases

    Returns None if unsupported or if fetching/parsing fails.
    """
    source_name = source["name"]
    url = source["base_url"]

    try:
        html = _fetch_html(url)
    except Exception:
        return None

    if source_name == "Federal Reserve FOMC":
        return _parse_fed_fomc(source=source, url=url, html=html, target_date=target_date)

    if source_name == "U.S. BLS CPI Releases":
        return _parse_bls_cpi(source=source, url=url, html=html, target_date=target_date)

    return None


def _fetch_html(url: str) -> str:
    """
    Fetch raw HTML text from a source URL.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    with httpx.Client(timeout=10.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _parse_fed_fomc(
    source: dict[str, Any],
    url: str,
    html: str,
    target_date: date,
) -> RawSourceItem | None:
    """
    Minimal parser for the Federal Reserve FOMC page.

    For now:
    - extract the HTML <title>
    - extract a short text snippet
    """
    title = _extract_title(html) or "Federal Reserve FOMC Update"
    snippet = _extract_snippet(html)

    return RawSourceItem(
        source_name=source["name"],
        source_type=source["source_type"],
        category=source["category"],
        url=url,
        headline=title,
        publish_date=target_date,
        content=snippet or "Federal Reserve FOMC page fetched successfully.",
    )


def _parse_bls_cpi(
    source: dict[str, Any],
    url: str,
    html: str,
    target_date: date,
) -> RawSourceItem | None:
    """
    Minimal parser for the BLS CPI page.

    For now:
    - extract the HTML <title>
    - extract a short text snippet
    """
    title = _extract_title(html) or "U.S. BLS CPI Update"
    snippet = _extract_snippet(html)

    return RawSourceItem(
        source_name=source["name"],
        source_type=source["source_type"],
        category=source["category"],
        url=url,
        headline=title,
        publish_date=target_date,
        content=snippet or "BLS CPI page fetched successfully.",
    )


def _extract_title(html: str) -> str | None:
    """
    Extract the page title from raw HTML.
    """
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return _strip_html_entities(title)


def _extract_snippet(html: str, max_length: int = 300) -> str:
    """
    Extract a rough plain-text snippet from HTML.

    This is intentionally lightweight for Step 11 and can be replaced
    later by BeautifulSoup or source-specific parsers.
    """
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _strip_html_entities(text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."
    return text


def _strip_html_entities(text: str) -> str:
    """
    Replace a few common HTML entities with plain-text equivalents.
    """
    replacements = {
        "&amp;": "&",
        "&nbsp;": " ",
        "&quot;": '"',
        "&#39;": "'",
        "&lt;": "<",
        "&gt;": ">",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


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