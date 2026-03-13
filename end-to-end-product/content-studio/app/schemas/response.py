"""
Response schemas for Content Studio API.
"""

from typing import Any

from pydantic import BaseModel, Field


class ScriptItem(BaseModel):
    """
    Script candidate or selected script.
    """

    style: str | None = None
    hook: str | None = None
    script: str | None = None
    cta: str | None = None


class StoryboardScene(BaseModel):
    """
    A single storyboard scene.
    """

    scene: int
    time_range: str
    visual: str
    voiceover: str
    on_screen_text: str


class StoryboardPackage(BaseModel):
    """
    Storyboard package output.
    """

    agent_name: str | None = None
    summary: str | None = None
    scenes: list[StoryboardScene] = Field(default_factory=list)
    editing_style: str | None = None


class PublishPackage(BaseModel):
    """
    Title/thumbnail/caption package output.
    """

    agent_name: str | None = None
    summary: str | None = None
    titles: list[str] = Field(default_factory=list)
    thumbnail_text: list[str] = Field(default_factory=list)
    caption: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    hook_reference: str | None = None
    angle_reference: str | None = None
    final_topic_reference: str | None = None


class MusicPackage(BaseModel):
    """
    Music direction package output.
    """

    agent_name: str | None = None
    summary: str | None = None
    bgm_direction: str | None = None
    suno_prompt: str | None = None
    editing_notes: list[str] = Field(default_factory=list)
    hook_reference: str | None = None
    final_topic_reference: str | None = None


class QAPackage(BaseModel):
    """
    QA selection result.
    """

    selected_script: int | None = None
    reason: str | None = None
    quality_score: float | None = None


class EditorBrief(BaseModel):
    """
    Editor-ready structured brief.
    """

    format: str | None = None
    platform: str | None = None
    duration_sec: int | None = None
    language: str | None = None
    topic: str | None = None
    audience: str | None = None
    tone: str | None = None
    hook: str | None = None
    narration_script: str | None = None
    cta: str | None = None
    scene_plan: list[StoryboardScene] = Field(default_factory=list)
    thumbnail_text: list[str] = Field(default_factory=list)
    editing_notes: list[str] = Field(default_factory=list)


class ContentGenerationResult(BaseModel):
    """
    Main result payload for content generation.
    """

    creative_brief: dict[str, Any] = Field(default_factory=dict)
    final_topic_suggestion: str | None = None
    script_candidates: list[ScriptItem] = Field(default_factory=list)
    best_script: ScriptItem | None = None
    revised_script: ScriptItem | None = None
    writer_outputs: dict[str, Any] = Field(default_factory=dict)
    storyboard_package: StoryboardPackage | None = None
    publish_package: PublishPackage | None = None
    music_package: MusicPackage | None = None
    qa_package: QAPackage | None = None
    editor_brief: EditorBrief | None = None
    video_generation_prompt: str | None = None


class ContentGenerationResponse(BaseModel):
    """
    Top-level API response schema.
    """

    request_id: str
    result: ContentGenerationResult