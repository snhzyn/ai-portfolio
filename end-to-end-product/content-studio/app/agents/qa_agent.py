from app.schemas.state import ContentStudioState


def qa_node(state: ContentStudioState) -> ContentStudioState:
    issues_found = []

    script = state.get("script_output", {})
    if len(script.get("hook_options", [])) < 2:
        issues_found.append("Not enough hook variation.")

    output = {
        "agent_name": "qa",
        "summary": "Basic QA review",
        "quality_score": 8.5 if not issues_found else 7.2,
        "issues_found": issues_found,
        "revision_notes": [
            "Ensure the first 3 seconds are visually strong.",
            "Keep on-screen text short and easy to scan.",
            "Make the CTA feel native to the platform.",
        ],
    }

    return {
        "qa_output": output,
        "logs": state.get("logs", []) + [{"node": "qa", "status": "completed"}],
    }