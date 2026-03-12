from app.schemas.state import ContentStudioState


def packaging_node(state: ContentStudioState) -> ContentStudioState:
    final_json = {
        "creative_brief": state.get("director_brief", {}),
        "research_package": state.get("research_output", {}),
        "script_package": state.get("script_output", {}),
        "storyboard_package": state.get("storyboard_output", {}),
        "publish_package": state.get("title_thumbnail_output", {}),
        "music_package": state.get("music_output", {}),
        "qa_package": state.get("qa_output", {}),
    }

    return {
        "final_json": final_json,
        "logs": state.get("logs", []) + [{"node": "packaging", "status": "completed"}],
    }