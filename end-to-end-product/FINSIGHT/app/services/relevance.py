"""
Country relevance scoring module.

This module maps normalized financial events to country-specific relevance
using profile-based transmission channels rather than simple geographic proximity.
"""

from __future__ import annotations

from app.config.country_profiles import COUNTRY_PROFILES
from app.schemas.event import EventItem


def estimate_country_relevance(event: EventItem, country: str) -> int:
    """
    Estimate how relevant an event is for the selected country.

    Scoring dimensions:
    - Direct country mention in region tags
    - Macro-financial sensitivity
    - Trade / supply-chain linkage
    - Sector exposure
    - Geopolitical sensitivity
    - Asset focus

    Returns an integer score from 1 to 10.
    """

    profile = COUNTRY_PROFILES.get(country)
    if not profile:
        return _fallback_relevance(event, country)

    score = 1

    region_tags = {tag.lower() for tag in event.region_tags}
    asset_tags = {tag.lower() for tag in event.asset_tags}
    headline = event.headline.lower()
    summary = event.summary.lower()
    market_impact = event.market_impact.lower()
    event_type = event.event_type.lower()

    text_blob = " ".join([headline, summary, market_impact, event_type])

    country_lower = country.lower()
    if country_lower in region_tags:
        score += 3

    if "global" in region_tags:
        score += 1

    # Macro-financial transmission
    macro_hits = _count_keyword_hits(profile.get("macro_sensitivity", []), asset_tags, text_blob)
    score += min(macro_hits, 2)

    # Trade / supply-chain linkage
    trade_hits = _count_keyword_hits(profile.get("trade_links", []), region_tags, text_blob)
    score += min(trade_hits, 2)

    # Sector exposure
    sector_hits = _count_keyword_hits(profile.get("key_sectors", []), asset_tags, text_blob)
    score += min(sector_hits, 2)

    # Geopolitical sensitivity
    geo_hits = _count_keyword_hits(profile.get("geo_risks", []), region_tags, text_blob)
    score += min(geo_hits, 2)

    # Asset focus
    asset_focus_hits = _count_keyword_hits(profile.get("asset_focus", []), asset_tags, text_blob)
    score += min(asset_focus_hits, 2)

    return max(1, min(score, 10))


def _count_keyword_hits(keywords: list[str], tag_set: set[str], text_blob: str) -> int:
    """
    Count how many profile keywords appear in either tags or free text.
    """

    hits = 0

    for keyword in keywords:
        keyword_lower = keyword.lower()
        normalized_keyword = keyword_lower.replace("_", " ")

        if keyword_lower in tag_set or keyword_lower in text_blob or normalized_keyword in text_blob:
            hits += 1

    return hits


def _fallback_relevance(event: EventItem, country: str) -> int:
    """
    Fallback heuristic if a country profile is not defined.
    """
    
    tags_lower = {tag.lower() for tag in event.region_tags + event.asset_tags}
    country_lower = country.lower()

    if country_lower in tags_lower:
        return 8
    if "global" in tags_lower:
        return 6
    return 5