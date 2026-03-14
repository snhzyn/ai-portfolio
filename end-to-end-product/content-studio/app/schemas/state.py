"""
Shared LangGraph state schema for Content Studio.
"""

import operator
from typing import Annotated, Any
from typing_extensions import NotRequired, TypedDict


class ContentStudioState(TypedDict, total=False):
    """
    Shared state object passed between LangGraph agents.
    """

    # request-level data
    request_id: str
    request: dict[str, Any]

    # director / planning
    director_brief: dict[str, Any]
    planned_agents: list[str]

    # research
    research_output: dict[str, Any]

    # writer outputs
    writer_fast_output: dict[str, Any] | None
    writer_story_output: dict[str, Any] | None
    writer_viral_output: dict[str, Any] | None

    # combined script candidate list
    script_candidates: Annotated[list[dict[str, Any]], operator.add]

    # selection / revision
    best_script: dict[str, Any] | None
    revised_script: dict[str, Any] | None
    qa_output: dict[str, Any] | None

    # downstream packages
    storyboard_output: dict[str, Any] | None
    title_thumbnail_output: dict[str, Any] | None
    music_output: dict[str, Any] | None

    # final topic normalization / refinement
    final_topic_suggestion: str | None

    # final packaging
    final_json: dict[str, Any] | None