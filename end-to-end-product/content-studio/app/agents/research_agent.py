from app.schemas.state import ContentStudioState


def research_node(state: ContentStudioState) -> ContentStudioState:
    topic = state["request"]["topic"]

    output = {
        "agent_name": "research",
        "summary": f"Key talking points for {topic}",
        "key_points": [
            f"{topic} is relevant because it has strong audience curiosity.",
            f"There are likely cultural, social, or practical reasons behind interest in {topic}.",
            f"The content should frame {topic} in a simple, memorable way.",
        ],
        "viewer_takeaway": f"After watching, viewers should quickly understand why {topic} matters.",
    }

    return {
        "research_output": output,
        "logs": state.get("logs", []) + [{"node": "research", "status": "completed"}],
    }