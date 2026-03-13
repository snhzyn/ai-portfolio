"""
Main FastAPI application entrypoint.
"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Content Studio API",
    description=(
        "A multi-agent content production API that generates short-form video "
        "packages including scripts, storyboard, publishing assets, editor brief, "
        "and video-generation-ready prompts."
    ),
    version="1.0.0",
    docs_url="/playground",
    redoc_url="/redoc",
)

app.include_router(router)