"""
Normalization helper.

These functions convert raw source items into normalized EventItem objects.
This keeps worker nodes focused on orchestration while moving event-shaping
logic into a dedicated service layer.
"""

from __future__ import annotations

from app.schemas.event import EventItem
from app.services.source_models import RawSourceItem


def normalize_macro_items(raw_items: list[RawSourceItem]) -> list[EventItem]:
    """
    Normalize macro-related raw items into EventItem objects.
    """

    return [
        EventItem(
            worker="macro",
            headline=item.headline,
            publish_date=item.publish_date,
            source=item.source_name,
            source_type=item.source_type,
            url=item.url,
            summary=item.content,
            market_impact="May affect global rates, USD direction, and risk-sensitive assets.",
            importance=7,
            confidence=0.85 if item.source_type == "official" else 0.75,
            region_tags=["US", "Global"],
            asset_tags=["rates", "usd", "equities"],
            event_type="macro_update",
        )
        for item in raw_items
    ]


def normalize_market_items(raw_items: list[RawSourceItem]) -> list[EventItem]:
    """
    Normalize market-related raw items into EventItem objects.
    """

    return [
        EventItem(
            worker="markets",
            headline=item.headline,
            publish_date=item.publish_date,
            source=item.source_name,
            source_type=item.source_type,
            url=item.url,
            summary=item.content,
            market_impact="May affect equity sentiment, large-cap technology names, and broader global risk appetite.",
            importance=7,
            confidence=0.82 if item.source_type in {"wire", "corporate"} else 0.72,
            region_tags=["US", "Global", "Korea"],
            asset_tags=["equities", "risk_sentiment", "semiconductors"],
            event_type="equity_market_update",
        )
        for item in raw_items
    ]


def normalize_commodities_fx_items(raw_items: list[RawSourceItem]) -> list[EventItem]:
    """
    Normalize commodities/FX/rates raw items into EventItem objects.
    """

    return [
        EventItem(
            worker="commodities_fx",
            headline=item.headline,
            publish_date=item.publish_date,
            source=item.source_name,
            source_type=item.source_type,
            url=item.url,
            summary=item.content,
            market_impact="May influence oil prices, dollar direction, inflation expectations, and rate-sensitive assets.",
            importance=8,
            confidence=0.86 if item.source_type == "official" else 0.8,
            region_tags=["Global", "Middle East", "Korea"],
            asset_tags=["oil", "fx", "rates", "inflation", "equities"],
            event_type="commodities_fx_update",
        )
        for item in raw_items
    ]


def normalize_geopolitical_items(raw_items: list[RawSourceItem]) -> list[EventItem]:
    """
    Normalize geopolitical raw items into EventItem objects.
    """
    
    return [
        EventItem(
            worker="geopolitical",
            headline=item.headline,
            publish_date=item.publish_date,
            source=item.source_name,
            source_type=item.source_type,
            url=item.url,
            summary=item.content,
            market_impact="May affect trade flows, supply chains, export-sensitive sectors, and cross-border risk sentiment.",
            importance=7,
            confidence=0.84 if item.source_type in {"official", "wire", "international_org"} else 0.72,
            region_tags=["Global", "US", "China", "Taiwan", "Korea"],
            asset_tags=["trade", "equities", "supply_chain", "risk_sentiment"],
            event_type="geopolitical_risk_update",
        )
        for item in raw_items
    ]