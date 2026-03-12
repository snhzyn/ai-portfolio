from pydantic import BaseModel, Field
from typing import Optional


class ContentGenerateRequest(BaseModel):
    topic: str = Field(..., description="Main content topic")
    platform: str = Field(default="youtube_shorts")
    audience: str = Field(default="general audience")
    tone: str = Field(default="engaging, concise, platform-native")
    duration_sec: int = Field(default=30, ge=10, le=180)
    reference_text: Optional[str] = None


class ContentGenerateResponse(BaseModel):
    request_id: str
    result: dict