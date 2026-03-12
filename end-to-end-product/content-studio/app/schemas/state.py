from typing import Any, TypedDict


class ContentStudioState(TypedDict, total=False):
    """
    Global LangGraph state for the Content Studio multi-agent system.
    """

    request_id: str
    request: dict[str, Any]

    director_brief: dict[str, Any]
    planned_agents: list[str]

    research_output: dict[str, Any]

    writer_fast_output: dict[str, Any]
    writer_story_output: dict[str, Any]
    writer_viral_output: dict[str, Any]

    script_candidates: list[dict[str, Any]]
    best_script: dict[str, Any]
    revised_script: dict[str, Any]

    final_topic_suggestion: str

    storyboard_output: dict[str, Any]
    title_thumbnail_output: dict[str, Any]
    music_output: dict[str, Any]

    qa_output: dict[str, Any]
    final_json: dict[str, Any]