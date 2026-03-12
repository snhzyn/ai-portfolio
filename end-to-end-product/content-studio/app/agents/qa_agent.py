"""
QA agent.

Evaluates multiple script candidates and selects the strongest one.
"""

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_haiku
import json


def qa_node(state: ContentStudioState) -> ContentStudioState:
    """
    Evaluate script candidates and select the best one.
    """

    scripts = state.get("script_candidates", [])

    prompt = f"""
You are evaluating short-form video scripts.

Select the best script based on:
- hook strength
- clarity
- pacing
- platform suitability

Scripts:
{scripts}

Return JSON:

{{
 "best_index": 0,
 "reason": "..."
}}
"""

    response = generate_with_haiku(prompt)

    try:
        parsed = json.loads(response)
        idx = parsed["best_index"]
    except Exception:
        idx = 0

    best_script = scripts[idx]

    return {
        "best_script": best_script,
        "qa_output": {
            "selected_script": idx
        },
        "logs": state.get("logs", []) + [{"node": "qa", "status": "completed"}],
    }