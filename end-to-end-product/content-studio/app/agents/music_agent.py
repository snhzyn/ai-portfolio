from app.schemas.state import ContentStudioState


def music_node(state: ContentStudioState) -> ContentStudioState:
    topic = state["request"]["topic"]
    tone = state["request"]["tone"]

    output = {
        "agent_name": "music",
        "summary": "Generated BGM direction and Suno-style prompt",
        "bgm_direction": "upbeat, modern, stylish, short-form-friendly",
        "suno_prompt": (
            f"Create a short-form background track for a video about {topic}. "
            f"Tone: {tone}. "
            f"Make it modern, catchy, energetic, and suitable for a 30-second social video. "
            f"Keep it clean, punchy, and edit-friendly."
        ),
        "editing_notes": [
            "Use fast jump cuts",
            "Add bold center captions",
            "Punch in with zoom on the first hook line",
            "Keep pacing tight with visual changes every 2-3 seconds",
        ],
    }

    return {
        "music_output": output,
        "logs": state.get("logs", []) + [{"node": "music", "status": "completed"}],
    }