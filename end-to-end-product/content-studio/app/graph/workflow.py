"""
LangGraph workflow definition for Content Studio.
"""

from langgraph.graph import StateGraph, START, END

from app.schemas.state import ContentStudioState
from app.agents.director import director_node

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

    graph.add_node("qa", qa_node)
    graph.add_node("revision", revision_node)

    graph.add_node("storyboard", storyboard_node)
    graph.add_node("title_thumbnail", title_thumbnail_node)
    graph.add_node("music", music_node)

    graph.add_node("packaging", packaging_node)

    # Start with director
    graph.add_edge(START, "director")

    # Writers run after director
    graph.add_edge("director", "writer_fast")
    graph.add_edge("director", "writer_story")
    graph.add_edge("director", "writer_viral")

    # QA runs after all writers finish
    graph.add_edge("writer_fast", "qa")
    graph.add_edge("writer_story", "qa")
    graph.add_edge("writer_viral", "qa")

    # Revision runs after QA
    graph.add_edge("qa", "revision")

    # Asset-generation agents run only after the revised script exists
    graph.add_edge("revision", "storyboard")
    graph.add_edge("revision", "title_thumbnail")
    graph.add_edge("revision", "music")

    # Packaging runs after asset-generation agents
    graph.add_edge("storyboard", "packaging")
    graph.add_edge("title_thumbnail", "packaging")
    graph.add_edge("music", "packaging")

    graph.add_edge("packaging", END)

    return graph.compile()