"""
Raw source data models for FINSIGHT.

These models represent unprocessed items fetched from external sources
before they are normalized into EventItem objects.
"""

from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field


class RawSourceItem(BaseModel):
    """
    Raw item fetched from a configured source.

    This is an intermediate structure between external content retrieval
    and normalized financial event generation.
    """

    source_name: str = Field(..., description="Human-readable source name")
    source_type: str = Field(..., description="Source trust category")
    category: str = Field(..., description="Registry category such as monetary_policy or equities")
    url: str = Field(..., description="Original source URL")
    headline: str = Field(..., description="Raw title or headline")
    publish_date: date = Field(..., description="Publication date")
    content: str = Field(..., description="Short raw content snippet or summary")
