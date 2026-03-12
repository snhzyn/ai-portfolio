"""
Dynamic agent router.

This node converts the director's plan into LangGraph routing decisions.
"""

from langgraph.types import Send
from app.schemas.state import ContentStudioState


def route_agents(state: ContentStudioState):
    """
    Send execution to the planned agents.
    """

    planned = state.get("planned_agents", [])

    routes = []

    for agent in planned:
        routes.append(Send(agent, state))

    return routes