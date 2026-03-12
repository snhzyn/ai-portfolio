from app.schemas.state import ContentStudioState


def title_thumbnail_node(state: ContentStudioState) -> ContentStudioState:
    topic = state["request"]["topic"]

    output = {
        "agent_name": "title_thumbnail",
        "summary": "Generated title, thumbnail, caption package",
        "titles": [
            f"Why {topic} Is Suddenly Everywhere",
            f"The Real Reason {topic} Is Trending",
            f"What’s Actually Driving the {topic} Hype?",
        ],
        "thumbnail_text": [
            f"{topic.upper()} EXPLAINED",
            "WHY IT'S TRENDING",
            "MORE THAN JUST HYPE",
        ],
        "caption": f"{topic} is getting a lot of attention right now — here’s why it matters.",
        "hashtags": ["#trend", "#shorts", "#viralcontent", "#contentstudio"],
    }

    return {
        "title_thumbnail_output": output,
        "logs": state.get("logs", []) + [{"node": "title_thumbnail", "status": "completed"}],
    }