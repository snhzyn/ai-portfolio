"""
Main FastAPI application entrypoint.
"""

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/studio")
def studio_page(request: Request):
    """
    Render the user-facing studio page.
    """
    return templates.TemplateResponse(
        "studio.html",
        {"request": request},
    )


@app.get("/")
def root():
    return RedirectResponse(url="/studio")