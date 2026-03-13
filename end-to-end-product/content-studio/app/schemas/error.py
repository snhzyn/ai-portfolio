"""
Error response schemas for Content Studio API.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    """

    error: str = Field(..., description="Machine-readable error code.")
    detail: str = Field(..., description="Human-readable error detail.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "generation_failed",
                "detail": "Failed to generate content package due to upstream model error.",
            }
        }
    }