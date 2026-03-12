from app.schemas.state import ContentStudioState


def storyboard_node(state: ContentStudioState) -> ContentStudioState:
    topic = state["request"]["topic"]

    scenes = [
        {
            "scene": 1,
            "time_range": "0-3s",
            "visual": f"Strong opening visual related to {topic}",
            "voiceover": f"Why is everyone suddenly talking about {topic}?",
            "on_screen_text": f"{topic.upper()} IS EVERYWHERE",
        },
        {
            "scene": 2,
            "time_range": "3-10s",
            "visual": "Fast montage of examples or trend signals",
            "voiceover": "It’s not random. There are clear reasons this is getting attention.",
            "on_screen_text": "WHY NOW?",
        },
        {
            "scene": 3,
            "time_range": "10-22s",
            "visual": "Explain 2-3 simple reasons with text overlays",
            "voiceover": "It’s attention-grabbing, culturally relevant, and easy to share.",
            "on_screen_text": "ATTENTION + RELEVANCE + SHAREABILITY",
        },
        {
            "scene": 4,
            "time_range": "22-30s",
            "visual": "End screen / reaction shot / CTA frame",
            "voiceover": "So is this just hype, or the start of something bigger?",
            "on_screen_text": "TREND OR SHIFT?",
        },
    ]

    return {
        "storyboard_output": {
            "agent_name": "storyboard",
            "summary": "Generated storyboard and shot plan",
            "scenes": scenes,
        },
        "logs": state.get("logs", []) + [{"node": "storyboard", "status": "completed"}],
    }