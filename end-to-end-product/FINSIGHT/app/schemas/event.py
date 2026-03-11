
"""
This module defines the core data structures used to represent financial events that
discovered by worker agents and later analyzed by the Lead Analyst.

The event lifecycle is as follows:

Worker Agents
    ↓
EventItem (raw structured event)
    ↓
Lead Analyst Agent
    ↓
RankedEventItem (prioritized event)

These schemas are shared across the entire pipeline:
- Worker agents
- Analyst agent
- Report generation
- API responses
"""


from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    """
    Trust category of the information source.

    This classification is used when computing confidence scores
    and verification status during analyst review.
    """
    OFFICIAL = "official"
    WIRE = "wire"
    MEDIA = "media"
    CORPORATE = "corporate"
    INTERNATIONAL_ORG = "international_org"

class VerificationStatus(str, Enum):
    """
    Verification result assigned by the Lead Analyst agent.

    - VERIFIED: Multiple trusted sources confirm the event.
    - PARTIALLY_VERIFIED: Some confirmation exists but evidence is limited.
    - UNVERIFIED: Only one weak source or insufficient evidence.
    - CONFLICTING: Sources disagree on key facts.
    """
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONFLICTING = "conflicting"

class WorkerName(str, Enum):
    MACRO = "macro"
    MARKETS = "markets"
    COMMODITIES_FX = "commodities_fx"
    GEOPOLITICAL = "geopolitical"

class EventItem(BaseModel):
    """
    Raw financial event produced by a worker agent.

    This represents the structured output extracted from various sources.

    These events are later processed by the Lead Analyst agent to perform:
    - Deduplication
    - Verification
    - Country relevance scoring
    - Priority ranking
    """
    worker: WorkerName = Field(..., description="Worker that produced this event item.")
    headline: str = Field(..., min_length=5, max_length=300)
    publish_date: date = Field(..., description="Publish date of the source item.")
    source: str = Field(..., min_length=2, max_length=100)
    source_type: SourceType = Field(..., description="Trust category of the source.")
    url: str = Field(..., description="Original source URL.")
    summary: str = Field(..., min_length=20, max_length=1200)
    market_impact: str = Field(..., min_length=10, max_length=800)
    importance: int = Field(..., ge=1, le=10, description="Global market importance, scale 1-10.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Reliability/confidence score, 0.0-1.0.")
    region_tags: list[str] = Field(default_factory=list, description="Relevant regions/countries, e.g. US, Korea, Global.")
    asset_tags: list[str] = Field(default_factory=list, description="Relevant assets, e.g. equities, fx, oil, rates.")
    event_type: str = Field(..., min_length=2, max_length=100, description="Normalized event category.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "worker": "macro",
                "headline": "Fed signals rates may remain higher for longer",
                "publish_date": "2026-03-10",
                "source": "Federal Reserve",
                "source_type": "official",
                "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                "summary": "The Fed indicated that inflation remains sticky and policy easing may be delayed.",
                "market_impact": "Supports USD and Treasury yields; negative for rate-sensitive equities.",
                "importance": 8,
                "confidence": 0.91,
                "region_tags": ["US", "Global"],
                "asset_tags": ["rates", "usd", "equities"],
                "event_type": "monetary_policy"
            }
        }
    }

class RankedEventItem(EventItem):
    """
    Financial event after Lead Analyst processing.

    The Lead Analyst agent evaluates all worker outputs and produces a final ranked set of events based on:
    - verification status
    - global importance
    - country-specific relevance
    """
    verification_status: VerificationStatus = Field(..., description="Verification result assigned by Lead Analyst.")
    country_relevance: int = Field(..., ge=1, le=10, description="Country-specific relevance, scale 1-10.")
    priority_score: float = Field(..., ge=0.0, le=10.0, description="Final ranking score.")
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="Additional supporting source names used for clustering/verification."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "worker": "commodities_fx",
                "headline": "Oil prices rise on renewed Middle East tensions",
                "publish_date": "2026-03-10",
                "source": "Reuters",
                "source_type": "wire",
                "url": "https://www.reuters.com/markets/commodities/",
                "summary": "Oil prices climbed as renewed tensions increased supply disruption concerns.",
                "market_impact": "Higher energy prices may pressure importers and transport-heavy sectors.",
                "importance": 8,
                "confidence": 0.87,
                "region_tags": ["Middle East", "Global", "Korea"],
                "asset_tags": ["oil", "fx", "equities"],
                "event_type": "energy_supply_risk",
                "verification_status": "verified",
                "country_relevance": 9,
                "priority_score": 8.5,
                "supporting_sources": ["EIA Weekly Petroleum Status Report"]
            }
        }
    }

class WorkerOutput(BaseModel):
    """
    Output container for a worker agent.

    Each worker returns a list of structured events that match
    the manager's briefing scope.
    """
    worker: WorkerName
    events: list[EventItem] = Field(default_factory=list)