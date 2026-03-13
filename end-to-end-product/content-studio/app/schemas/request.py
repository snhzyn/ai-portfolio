"""
Request schema for Content Studio.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ContentGenerationRequest(BaseModel):
    """
    Request payload for generating a short-form content package.
    """

    topic: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Main topic or question for the short-form content.",
        examples=["왜 요즘 러닝이 MZ세대 사이에서 다시 유행하는가"],
    )

    platform: Literal["youtube_shorts", "tiktok", "instagram_reels"] = Field(
        ...,
        description="Target short-form platform.",
        examples=["youtube_shorts"],
    )

    audience: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Target audience for the content.",
        examples=["10~30대 직장인과 MZ세대"],
    )

    tone: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Desired tone and style.",
        examples=["재밌고 빠른 템포"],
    )

    duration_sec: Literal[15, 30, 45, 60] = Field(
        ...,
        description="Target video duration in seconds.",
        examples=[30],
    )

    reference_text: str = Field(
        default="",
        max_length=1000,
        description="Optional additional guidance or context.",
        examples=["고등학생이 이해할 수 있는 수준으로, 마지막에 짧은 영향도 설명"],
    )

    language: Literal["ko", "en"] = Field(
        ...,
        description="Output language.",
        examples=["ko"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "왜 요즘 러닝이 MZ세대 사이에서 다시 유행하는가",
                "platform": "youtube_shorts",
                "audience": "10~30대 직장인과 MZ세대",
                "tone": "재밌고 빠른 템포",
                "duration_sec": 30,
                "reference_text": "운동, 커뮤니티, 자기관리 관점도 넣어줘.",
                "language": "ko",
            }
        }
    }