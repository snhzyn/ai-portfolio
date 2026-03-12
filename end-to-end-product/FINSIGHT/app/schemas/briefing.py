
"""
This module defines the request and response models used by the
FastAPI endpoint that generates daily financial intelligence reports.

Pipeline overview:

User Request
    ↓
Manager Agent
    ↓
Worker Agents
    ↓
Lead Analyst Agent
    ↓
Report Agent
    ↓
BriefingResponse
"""


from __future__ import annotations

from datetime import date as DateType
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.event import RankedEventItem, WorkerOutput


class BriefingRequest(BaseModel):
    """
    Request payload for generating a financial intelligence briefing.
    """
    date: DateType = Field(..., description="Target date for the briefing in YYYY-MM-DD format.")
    country: str = Field(..., min_length=2, max_length=100, description="Country focus, e.g. Korea.")
    max_items_per_worker: int = Field(default=5, ge=1, le=10)
    report_type: Literal["eod"] = Field(default="eod", description="Report type. v1 supports end-of-day only.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2026-03-10",
                "country": "Korea",
                "max_items_per_worker": 5,
                "report_type": "eod"
            }
        }
    }

class BriefingMeta(BaseModel):
    """
    Metadata describing how the briefing was generated.
    """
    workers_run: int = Field(default=4)
    events_collected: int = Field(default=0)
    events_after_dedup: int = Field(default=0)
    generated_for_country: str
    cutoff_date: DateType

class BriefingResponse(BaseModel):
    """
    Final response returned by the FastAPI endpoint.

    This response contains:

    - human-readable market briefing
    - structured event list
    - generation metadata
    """
    date: DateType
    country: str
    report_markdown: str
    top_events: list[RankedEventItem] = Field(default_factory=list)
    meta: BriefingMeta

class BriefingDebugResponse(BaseModel):
    """
    Debug response used for development and testing.

    Exposes intermediate agent outputs for inspection.
    """
    date: DateType
    country: str
    manager_plan: dict[str, Any]
    worker_outputs: dict[str, WorkerOutput]
    analyst_output: dict[str, Any]
    report_markdown: str