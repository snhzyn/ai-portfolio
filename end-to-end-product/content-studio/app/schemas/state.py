from typing import Any, TypedDict


class ContentStudioState(TypedDict, total=False):
    """
    Global LangGraph state for the Content Studio multi-agent system.
    """

    request_id: str
    request: dict[str, Any]

    director_brief: dict[str, Any]

    # NEW
    planned_agents: list[str]

    research_output: dict[str, Any]
    script_output: dict[str, Any]
    storyboard_output: dict[str, Any]
    title_thumbnail_output: dict[str, Any]
    music_output: dict[str, Any]

    qa_output: dict[str, Any]
    final_json: dict[str, Any]

    logs: list[dict[str, str]]