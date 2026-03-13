"""
API routes for Content Studio.
"""

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.graph.workflow import build_graph
from app.schemas import (
    ContentGenerationRequest,
    ContentGenerationResponse,
    ErrorResponse,
)

router = APIRouter(tags=["Content Studio"])


@router.post(
    "/api/content/generate",
    response_model=ContentGenerationResponse,
    summary="Generate a short-form content package",
    description=(
        "Generates a production-ready short-form content package including "
        "multiple script candidates, QA-selected script, revised script, "
        "storyboard, publishing assets, music prompt, editor brief, and a "
        "video-generation-ready prompt."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid request payload or business validation failure.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal generation failure.",
        },
    },
)
def generate_content(request: ContentGenerationRequest):
    """
    Generate a full short-form content package from the input request.
    """
    try:
        graph = build_graph()

        initial_state = {
            "request_id": str(uuid.uuid4()),
            "request": request.model_dump(),
        }

        result = graph.invoke(initial_state)
        final_json = result.get("final_json")

        if not final_json:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "generation_failed",
                    "detail": "The workflow completed without producing a final result.",
                },
            )

        return {
            "request_id": initial_state["request_id"],
            "result": final_json,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "detail": str(exc),
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "generation_failed",
                "detail": str(exc),
            },
        )