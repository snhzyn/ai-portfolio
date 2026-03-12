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


def macro_worker_node(state: FinanceBriefingState):
    """
    Dummy macro worker.

    Returns one macro event aligned with the selected date.
    """

    event = EventItem(
        worker="macro",
        headline="Fed signals rates may remain higher for longer",
        publish_date=state["date"],
        source="Federal Reserve",
        source_type="official",
        url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        summary="The Fed indicated that inflation remains sticky and policy easing may be delayed.",
        market_impact="Supports USD and Treasury yields; negative for rate-sensitive equities.",
        importance=8,
        confidence=0.91,
        region_tags=["US", "Global"],
        asset_tags=["rates", "usd", "equities"],
        event_type="monetary_policy",
    )

    return {
        "macro_events": [event],
    }


def markets_worker_node(state: FinanceBriefingState):
    """
    Dummy markets worker.

    Returns one market sentiment/equities event.
    """

    event = EventItem(
        worker="markets",
        headline="US tech stocks end mixed as investors reassess AI valuations",
        publish_date=state["date"],
        source="Reuters Markets",
        source_type="wire",
        url="https://www.reuters.com/markets/",
        summary="Large-cap technology stocks closed mixed as investors weighed valuation concerns against AI demand momentum.",
        market_impact="Mixed impact on global risk sentiment and semiconductor-related equities.",
        importance=7,
        confidence=0.84,
        region_tags=["US", "Global", "Korea"],
        asset_tags=["equities", "semiconductors", "risk_sentiment"],
        event_type="equity_market_sentiment",
    )

    return {
        "markets_events": [event],
    }


def commodities_fx_worker_node(state: FinanceBriefingState):
    """
    Dummy commodities/FX worker.

    Returns one energy-linked event.
    """

    event = EventItem(
        worker="commodities_fx",
        headline="Oil prices rise on renewed Middle East supply concerns",
        publish_date=state["date"],
        source="Reuters Commodities",
        source_type="wire",
        url="https://www.reuters.com/markets/commodities/",
        summary="Oil prices moved higher as markets priced in elevated geopolitical supply risk.",
        market_impact="Higher energy prices may pressure importers and transport-heavy sectors while supporting inflation concerns.",
        importance=8,
        confidence=0.87,
        region_tags=["Middle East", "Global", "Korea"],
        asset_tags=["oil", "fx", "equities", "inflation"],
        event_type="energy_supply_risk",
    )

    return {
        "commodities_fx_events": [event],
    }


def geopolitical_worker_node(state: FinanceBriefingState):
    """
    Dummy geopolitical worker.

    Returns one geopolitical risk event with financial spillover.
    """

    event = EventItem(
        worker="geopolitical",
        headline="New trade restrictions raise concerns over semiconductor supply chains",
        publish_date=state["date"],
        source="Reuters World",
        source_type="wire",
        url="https://www.reuters.com/world/",
        summary="New trade restrictions increased uncertainty around semiconductor supply chains and export flows.",
        market_impact="May weigh on semiconductor exporters and raise cross-border trade uncertainty.",
        importance=7,
        confidence=0.82,
        region_tags=["US", "China", "Taiwan", "Korea", "Global"],
        asset_tags=["equities", "semiconductors", "trade"],
        event_type="trade_restriction",
    )

    return {
        "geopolitical_events": [event],
    }


def lead_analyst_node(state: FinanceBriefingState):
    """
    Aggregate, deduplicate, and rank worker outputs.

    Simply,
    - merge all worker events
    - skip real deduplication for now
    - assign fixed verification/country relevance logic
    - compute priority score
    """
    
    all_events = (
        state.get("macro_events", [])
        + state.get("markets_events", [])
        + state.get("commodities_fx_events", [])
        + state.get("geopolitical_events", [])
    )

    country = state["country"]

    ranked_events: list[RankedEventItem] = []
    for event in all_events:
        country_relevance = _estimate_country_relevance(event=event, country=country)
        priority_score = round(
            (0.5 * event.importance) + (0.3 * country_relevance) + (0.2 * (event.confidence * 10)),
            2,
        )

        ranked_event = RankedEventItem(
            **event.model_dump(),
            verification_status="verified" if event.confidence >= 0.85 else "partially_verified",
            country_relevance=country_relevance,
            priority_score=priority_score,
            supporting_sources=[],
        )
        ranked_events.append(ranked_event)

    ranked_events.sort(key=lambda x: x.priority_score, reverse=True)

    analyst_summary = {
        "total_events_collected": len(all_events),
        "total_events_after_dedup": len(all_events),
        "top_themes": [event.event_type for event in ranked_events[:3]],
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
            "Step 4 uses dummy worker outputs.",
            "Deduplication and verification logic will be expanded in later steps.",
        ],
    }

    return {
        "all_events": all_events,
        "deduplicated_events": all_events,
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


def _estimate_country_relevance(event: EventItem, country: str) -> int:
    """
    Simple heuristic for country relevance.

    This will be replaced by profile-based scoring later.
    """

    tags_lower = {tag.lower() for tag in event.region_tags + event.asset_tags}
    country_lower = country.lower()

    if country_lower in tags_lower:
        return 9
    if "global" in tags_lower:
        return 7
    if country == "Korea" and any(tag in tags_lower for tag in ["oil", "semiconductors", "usd", "rates"]):
        return 8
    return 5


def _bullet_list(items: list[str]) -> str:
    """
    Render a list of strings as markdown bullets.
    """
    
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)