"""
Revision agent.

Improves the selected script using QA feedback.
"""

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
import json


def revision_node(state: ContentStudioState) -> ContentStudioState:
    """
    Improve the selected script based on QA feedback.
    """

    script = state.get("best_script", {})
    topic = state["request"]["topic"]

    prompt = f"""
You are improving a short-form video script.

Topic: {topic}

Original script:
{script}

Improve the script by:
- making the hook stronger
- tightening pacing
- making the CTA clearer

Return JSON:

{{
 "hook": "...",
 "script": "...",
 "cta": "..."
}}
"""

    response = generate_with_sonnet(prompt)

    try:
        revised = json.loads(response)
    except Exception:
        revised = script

    return {
        "revised_script": revised,
        "logs": state.get("logs", []) + [{"node": "revision", "status": "completed"}],
    }