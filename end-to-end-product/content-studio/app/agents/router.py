"""
Routing utilities for the LangGraph workflow.
"""

from app.schemas.state import ContentStudioState


_REQUIRED_AGENTS = [
    "writer_fast",
    "writer_story",
    "writer_viral",
    "storyboard",
    "title_thumbnail",
    "music",
]


def route_agents(state: ContentStudioState) -> list[str]:
    """
    Route from the research stage to downstream specialist agents.

    Writers and downstream production agents are always included to keep the
    multi-agent pipeline stable and complete.

    Args:
        state: The current LangGraph state.

    Returns:
        List of downstream node names.
    """
    planned_agents = state.get("planned_agents") or []

    valid_agents = set(_REQUIRED_AGENTS)

    routed = []
    seen = set()

    for agent in planned_agents:
        if agent in valid_agents and agent not in seen:
            routed.append(agent)
            seen.add(agent)

    for agent in _REQUIRED_AGENTS:
        if agent not in seen:
            routed.append(agent)
            seen.add(agent)

    return routed