"""
Routing helpers for dynamic agent execution.

This module contains conditional edge routing functions used by LangGraph
to dynamically dispatch specialist agents based on the director's plan.
"""

from langgraph.types import Send

from app.schemas.state import ContentStudioState


def route_agents(state: ContentStudioState):
    """
    Route execution to the specialist agents selected by the director.

    Args:
        state: The current graph state containing the director's planned agents.

    Returns:
        A list of Send instructions for LangGraph conditional routing.
    """
    planned = state.get("planned_agents", [])
    return [Send(agent_name, state) for agent_name in planned]