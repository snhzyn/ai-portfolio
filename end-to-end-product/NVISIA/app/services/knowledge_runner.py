from app.services.knowledge import KnowledgeGraph

def update_knowledge_graph_state(rec_list, selected_id, session_state):
    """
    Update session_state with the latest knowledge graph figure.
    The graph is regenerated only when the selected article changes.
    """

    if selected_id != session_state["last_selected_id_for_kg"]:
        session_state["knowledge_fig"] = None
        session_state["kg_error"] = None

        if rec_list:
            try:
                know = KnowledgeGraph(rec_list)
                fig = know.get_figure()
                session_state["knowledge_fig"] = fig
            except Exception as e:
                session_state["kg_error"] = str(e)

        session_state["last_selected_id_for_kg"] = selected_id