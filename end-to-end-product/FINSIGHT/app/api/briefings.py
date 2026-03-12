"""
FastAPI routes for FINSIGHT briefing generation.

This module exposes API endpoints that run the LangGraph-based
financial intelligence workflow.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.graph.workflow import build_workflow
from app.schemas.briefing import (
    BriefingDebugResponse,
    BriefingMeta,
    BriefingRequest,
    BriefingResponse,
)
from app.schemas.event import WorkerOutput


router = APIRouter(prefix="/api/v1/briefings", tags=["briefings"])


@router.post("/run", response_model=BriefingResponse)
def run_briefing(request: BriefingRequest) -> BriefingResponse:
    """
    Run the full FINSIGHT workflow and return a human-readable report
    plus the ranked top events used to generate it.
    """

    workflow = build_workflow()

    initial_state = {
        "date": request.date,
        "country": request.country,
        "max_items_per_worker": request.max_items_per_worker,
        "report_type": request.report_type,
        "logs": [],
        "errors": [],
    }

    try:
        result = workflow.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}") from exc

    ranked_events = result.get("ranked_events", [])
    all_events = result.get("all_events", [])
    deduplicated_events = result.get("deduplicated_events", [])

    meta = BriefingMeta(
        workers_run=4,
        events_collected=len(all_events),
        events_after_dedup=len(deduplicated_events),
        generated_for_country=request.country,
        cutoff_date=request.date,
    )

    return BriefingResponse(
        date=request.date,
        country=request.country,
        report_markdown=result.get("report_markdown", ""),
        top_events=ranked_events,
        meta=meta,
    )


@router.post("/debug", response_model=BriefingDebugResponse)
def debug_briefing(request: BriefingRequest) -> BriefingDebugResponse:
    """
    Run the workflow and expose intermediate outputs for debugging.

    Useful during development to inspect:
    - manager plan
    - worker outputs
    - analyst output
    """
    workflow = build_workflow()

    initial_state = {
        "date": request.date,
        "country": request.country,
        "max_items_per_worker": request.max_items_per_worker,
        "report_type": request.report_type,
        "logs": [],
        "errors": [],
    }

    try:
        result = workflow.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}") from exc

    worker_outputs = {
        "macro": WorkerOutput(worker="macro", events=result.get("macro_events", [])),
        "markets": WorkerOutput(worker="markets", events=result.get("markets_events", [])),
        "commodities_fx": WorkerOutput(
            worker="commodities_fx",
            events=result.get("commodities_fx_events", []),
        ),
        "geopolitical": WorkerOutput(
            worker="geopolitical",
            events=result.get("geopolitical_events", []),
        ),
    }

    analyst_output = {
        "all_events": [event.model_dump() for event in result.get("all_events", [])],
        "deduplicated_events": [
            event.model_dump() for event in result.get("deduplicated_events", [])
        ],
        "ranked_events": [event.model_dump() for event in result.get("ranked_events", [])],
        "analyst_summary": result.get("analyst_summary", {}),
        "logs": result.get("logs", []),
        "errors": result.get("errors", []),
    }

    return BriefingDebugResponse(
        date=request.date,
        country=request.country,
        manager_plan=result.get("manager_plan", {}),
        worker_outputs=worker_outputs,
        analyst_output=analyst_output,
        report_markdown=result.get("report_markdown", ""),
    )