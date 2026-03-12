from app.schemas.state import ContentStudioState


def director_node(state: ContentStudioState) -> ContentStudioState:
    request = state["request"]

    brief = {
        "content_goal": f"Create a production-ready short-form video package about '{request['topic']}'.",
        "target_audience": request["audience"],
        "platform_strategy": f"Optimize for {request['platform']} with a strong opening hook and fast pacing.",
        "tone_guidance": request["tone"],
        "core_angle": f"Present '{request['topic']}' in a way that feels useful, engaging, and easy to turn into a short video.",
        "deliverables": [
            "hook options",
            "short script",
            "storyboard",
            "title and thumbnail copy",
            "caption and hashtags",
            "music prompt",
            "editing notes",
        ],
    }

    return {
        "director_brief": brief,
        "logs": state.get("logs", []) + [{"node": "director", "status": "completed"}],
    }