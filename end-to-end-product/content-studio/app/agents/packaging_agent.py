"""
Packaging agent.

Combines outputs from all agents into the final API response payload.
"""

from app.schemas.state import ContentStudioState


def packaging_node(state: ContentStudioState) -> ContentStudioState:
    """
    Package all agent outputs into the final JSON structure.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the final JSON payload.
    """
    final_json = {
        "creative_brief": state.get("director_brief"),
        "script_candidates": state.get("script_candidates"),
        "best_script": state.get("best_script"),
        "revised_script": state.get("revised_script"),
        "writer_outputs": {
            "writer_fast": state.get("writer_fast_output"),
            "writer_story": state.get("writer_story_output"),
            "writer_viral": state.get("writer_viral_output"),
        },
        "storyboard_package": state.get("storyboard_output"),
        "publish_package": state.get("title_thumbnail_output"),
        "music_package": state.get("music_output"),
        "qa_package": state.get("qa_output"),
    }

    return {
        "final_json": final_json,
    }