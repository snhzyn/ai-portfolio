from app.schemas.state import ContentStudioState


def script_node(state: ContentStudioState) -> ContentStudioState:
    topic = state["request"]["topic"]
    duration = state["request"]["duration_sec"]

    output = {
        "agent_name": "script",
        "summary": "Generated short-form video script",
        "hook_options": [
            f"Why is everyone suddenly talking about {topic}?",
            f"Here’s the real reason {topic} is taking off.",
            f"{topic} looks simple, but there’s more going on than you think.",
        ],
        "final_script": (
            f"Why is everyone suddenly talking about {topic}? "
            f"Here’s what’s really happening. "
            f"First, it grabs attention fast. "
            f"Second, it connects to a bigger trend people already care about. "
            f"And third, it’s easy to turn into a shareable opinion. "
            f"So if you keep seeing {topic} everywhere, now you know why."
        ),
        "cta": "What do you think — trend, hype, or something bigger?",
        "duration_sec": duration,
    }

    return {
        "script_output": output,
        "logs": state.get("logs", []) + [{"node": "script", "status": "completed"}],
    }