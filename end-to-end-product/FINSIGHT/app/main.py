"""
FastAPI application entrypoint for FINSIGHT.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.briefings import router as briefing_router


app = FastAPI(
    title="FINSIGHT",
    version="0.1.0",
    description="Multi-agent financial intelligence system built with FastAPI and LangGraph.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "FINSIGHT API is running. Visit /docs for interactive API documentation."
    }


app.include_router(briefing_router)