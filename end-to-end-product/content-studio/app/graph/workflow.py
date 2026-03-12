"""
LangGraph workflow definition for Content Studio.
"""

from langgraph.graph import StateGraph, START, END

from app.schemas.state import ContentStudioState
from app.agents.director import director_node
from app.agents.router import route_agents

from app.agents.writer_fast_agent import writer_fast_node
from app.agents.writer_story_agent import writer_story_node
from app.agents.writer_viral_agent import writer_viral_node

from app.agents.storyboard_agent import storyboard_node
from app.agents.title_thumbnail_agent import title_thumbnail_node
from app.agents.music_agent import music_node

from app.agents.qa_agent import qa_node
from app.agents.revision_agent import revision_node
from app.agents.packaging_agent import packaging_node


def build_graph():
    """
    Build and compile the LangGraph workflow for Content Studio.

    Returns:
        A compiled LangGraph application.
    """
    graph = StateGraph(ContentStudioState)

    graph.add_node("director", director_node)

    graph.add_node("writer_fast", writer_fast_node)
    graph.add_node("writer_story", writer_story_node)
    graph.add_node("writer_viral", writer_viral_node)

    graph.add_node("storyboard", storyboard_node)
    graph.add_node("title_thumbnail", title_thumbnail_node)
    graph.add_node("music", music_node)

    graph.add_node("qa", qa_node)
    graph.add_node("revision", revision_node)
    graph.add_node("packaging", packaging_node)

    graph.add_edge(START, "director")
    graph.add_conditional_edges("director", route_agents)

    graph.add_edge("writer_fast", "qa")
    graph.add_edge("writer_story", "qa")
    graph.add_edge("writer_viral", "qa")

    graph.add_edge("qa", "revision")
    graph.add_edge("revision", "packaging")
    graph.add_edge("packaging", END)

    return graph.compile()