"""
Graph node implementations for the FINSIGHT workflow.

This module contains the node functions used by LangGraph.

As a simple test, these nodes return deterministic dummy data so that
the end-to-end workflow can be tested before integrating real
source retrieval and Claude-based agent logic.
"""

from __future__ import annotations

from datetime import datetime

from app.graph.state import FinanceBriefingState
from app.schemas.event import EventItem, RankedEventItem

from app.services.relevance import estimate_country_relevance
from app.services.dedup import deduplicate_events
from app.services.normalizers import (
    normalize_commodities_fx_items,
    normalize_geopolitical_items,
    normalize_macro_items,
    normalize_market_items,
)
from app.services.source_fetcher import fetch_worker_sources


def manager_node(state: FinanceBriefingState):
    """
    Create a manager plan from the user's request.

    This node builds a simple deterministic plan that downstream
    workers can follow. 
    
    Later, this can be replaced with an LLM-powered planning agent.
    """

    target_date = state["date"]
    country = state["country"]
    report_type = state.get("report_type", "eod")

    manager_plan = {
        "date_cutoff": f"{target_date.isoformat()}T23:59:59",
        "country_focus": country,
        "report_type": report_type,
        "briefing_goal": f"Generate a daily finance intelligence report for {country}.",
        "worker_tasks": {
            "macro": "Collect top macro-financial developments relevant to the selected date and country.",
            "markets": "Collect top equity-market and risk-sentiment developments.",
            "commodities_fx": "Collect top commodities, FX, and rates developments.",
            "geopolitical": "Collect top geopolitical events with financial relevance.",
        },
        "source_constraints": {
            "macro": ["official", "wire"],
            "markets": ["wire", "corporate", "media"],
            "commodities_fx": ["official", "wire"],
            "geopolitical": ["official", "wire", "international_org"],
        },
        "notes": [
            "Exclude events published after the selected date.",
            "Prioritize trusted official and wire sources.",
        ],
    }

    return {
        "manager_plan": manager_plan,
        "logs": state.get("logs", []) + [{"node": "manager", "status": "completed"}],
    }


def macro_worker_node(state: FinanceBriefingState) -> FinanceBriefingState:
    """
    Macro worker using the source retrieval abstraction.
    """

    raw_items = fetch_worker_sources("macro", state["date"])
    events = normalize_macro_items(raw_items)

    return {
        "macro_events": events,
    }


def markets_worker_node(state: FinanceBriefingState) -> FinanceBriefingState:
    """
    Markets worker using the source retrieval abstraction.
    """
    raw_items = fetch_worker_sources("markets", state["date"])
    events = normalize_market_items(raw_items)

    return {
        "markets_events": events,
    }


def commodities_fx_worker_node(state: FinanceBriefingState) -> FinanceBriefingState:
    """
    Commodities/FX worker using the source retrieval abstraction.
    """
    raw_items = fetch_worker_sources("commodities_fx", state["date"])
    events = normalize_commodities_fx_items(raw_items)

    return {
        "commodities_fx_events": events,
    }


def geopolitical_worker_node(state: FinanceBriefingState) -> FinanceBriefingState:
    """
    Geopolitical risk worker using the source retrieval abstraction.
    """
    raw_items = fetch_worker_sources("geopolitical", state["date"])
    events = normalize_geopolitical_items(raw_items)

    return {
        "geopolitical_events": events,
    }


def lead_analyst_node(state: FinanceBriefingState) -> FinanceBriefingState:
    """
    Aggregate, deduplicate, and rank worker outputs.
    """
    all_events = (
        state.get("macro_events", [])
        + state.get("markets_events", [])
        + state.get("commodities_fx_events", [])
        + state.get("geopolitical_events", [])
    )

    deduplicated_events, supporting_sources_map = deduplicate_events(all_events)

    country = state["country"]

    ranked_events: list[RankedEventItem] = []
    for event in deduplicated_events:
        country_relevance = estimate_country_relevance(event=event, country=country)
        priority_score = round(
            (0.5 * event.importance) + (0.3 * country_relevance) + (0.2 * (event.confidence * 10)),
            2,
        )

        ranked_event = RankedEventItem(
            **event.model_dump(),
            verification_status="verified" if event.confidence >= 0.85 else "partially_verified",
            country_relevance=country_relevance,
            priority_score=priority_score,
            supporting_sources=supporting_sources_map.get(event.headline, []),
        )
        ranked_events.append(ranked_event)

    ranked_events.sort(key=lambda x: x.priority_score, reverse=True)

    top_themes = list(dict.fromkeys(event.event_type for event in ranked_events))[:3]

    analyst_summary = {
        "total_events_collected": len(all_events),
        "total_events_after_dedup": len(deduplicated_events),
        "top_themes": top_themes,
        "key_risks": [
            "Dollar strength",
            "Energy import pressure",
            "Trade-related semiconductor uncertainty",
        ],
        "cross_market_implications": [
            f"{country} may face tighter external financial conditions if USD and yields stay elevated.",
            "Energy-sensitive sectors should be monitored if oil remains high.",
        ],
        "watchlist": [
            "USD",
            "Oil",
            "Semiconductors",
            "Export-sensitive equities",
        ],
        "methodology_notes": [
            "This version uses source-registry-based placeholder retrieval and normalization logic.",
            "Simple event clustering is applied to reduce duplicate signals before ranking.",
            "Real source fetching, parsing, stronger deduplication, and verification will be expanded in later steps.",
        ],
    }

    return {
        "all_events": all_events,
        "deduplicated_events": deduplicated_events,
        "ranked_events": ranked_events,
        "analyst_summary": analyst_summary,
        "logs": state.get("logs", []) + [{"node": "lead_analyst", "status": "completed"}],
    }


def report_node(state: FinanceBriefingState):
    """
    Generate a human-readable markdown report from ranked events.

    For simple test, this uses simple string composition rather than an LLM.
    """
    target_date = state["date"].isoformat()
    country = state["country"]
    ranked_events = state.get("ranked_events", [])
    analyst_summary = state.get("analyst_summary", {})

    top_events_md = []
    for i, event in enumerate(ranked_events[:5], start=1):
        top_events_md.append(
            (
                f"### {i}. {event.headline}\n"
                f"- Source: {event.source}\n"
                f"- Verification: {event.verification_status.value}\n"
                f"- Importance: {event.importance}/10\n"
                f"- Country Relevance: {event.country_relevance}/10\n"
                f"- Priority Score: {event.priority_score}\n"
                f"- Why it matters: {event.market_impact}\n"
            )
        )

    report_markdown = f"""
# Daily Finance Intelligence Report  

## Report Profile  

**Date:** {target_date}  
**Country Focus:** {country}  

## Executive Summary  

This report summarizes the most relevant macro, market, commodities/FX, and geopolitical developments for {country} based on the selected date.

## Top Events  

{chr(10).join(top_events_md)}  

## Cross-Market Implications  
{_bullet_list(analyst_summary.get("cross_market_implications", []))}  

## Watchlist  
{_bullet_list(analyst_summary.get("watchlist", []))}  

## Methodology Notes  
{_bullet_list(analyst_summary.get("methodology_notes", []))}  
"""

    return {
        "report_markdown": report_markdown,
        "logs": state.get("logs", []) + [{"node": "report", "status": "completed"}],
    }


def _bullet_list(items: list[str]) -> str:
    """
    Render a list of strings as markdown bullets.
    """
    
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)