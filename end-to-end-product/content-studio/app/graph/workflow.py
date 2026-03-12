from langgraph.graph import StateGraph, START, END

from app.schemas.state import ContentStudioState
from app.agents.director import director_node
from app.agents.research_agent import research_node
from app.agents.script_agent import script_node
from app.agents.storyboard_agent import storyboard_node
from app.agents.title_thumbnail_agent import title_thumbnail_node
from app.agents.music_agent import music_node
from app.agents.qa_agent import qa_node
from app.agents.packaging_agent import packaging_node


def build_graph():
    graph = StateGraph(ContentStudioState)

    graph.add_node("director", director_node)
    graph.add_node("research", research_node)
    graph.add_node("script", script_node)
    graph.add_node("storyboard", storyboard_node)
    graph.add_node("title_thumbnail", title_thumbnail_node)
    graph.add_node("music", music_node)
    graph.add_node("qa", qa_node)
    graph.add_node("packaging", packaging_node)

    graph.add_edge(START, "director")

    graph.add_edge("director", "research")
    graph.add_edge("director", "script")
    graph.add_edge("director", "storyboard")
    graph.add_edge("director", "title_thumbnail")
    graph.add_edge("director", "music")

    graph.add_edge("research", "qa")
    graph.add_edge("script", "qa")
    graph.add_edge("storyboard", "qa")
    graph.add_edge("title_thumbnail", "qa")
    graph.add_edge("music", "qa")

    graph.add_edge("qa", "packaging")
    graph.add_edge("packaging", END)

    return graph.compile()