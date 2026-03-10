def init_session_state(session_state):
    defaults = {
        "page": "home",
        "uploaded_csv": None,
        "ingest_status": {"done": False, "msg": ""},
        "selected_id": None,
        "expanded": False,
        "last_selected_id_for_kg": None,
        "knowledge_fig": None,
        "kg_error": None,
    }

    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value