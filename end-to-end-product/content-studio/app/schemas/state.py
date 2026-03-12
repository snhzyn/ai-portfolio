from typing import Any, TypedDict


class ContentStudioState(TypedDict, total=False):
    """
    Global LangGraph state for the Content Studio multi-agent system.
    """

    request_id: str
    request: dict

    director_brief: dict
    planned_agents: list[str]

    research_output: dict

    script_candidates: list[dict]
    best_script: dict
    revised_script: dict

    storyboard_output: dict
    title_thumbnail_output: dict
    music_output: dict

    qa_output: dict
    final_json: dict

    logs: list[dict]