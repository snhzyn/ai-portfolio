"""
QA agent.

Evaluates multiple writer-generated script candidates and selects the strongest one.
"""

import json

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_haiku
from app.services.json_utils import parse_json_response


def qa_node(state: ContentStudioState) -> ContentStudioState:
    """
    Evaluate the available script candidates and select the best one.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the selected best script and QA metadata.
    """
    candidates = [
        state.get("writer_fast_output"),
        state.get("writer_story_output"),
        state.get("writer_viral_output"),
    ]
    candidates = [candidate for candidate in candidates if candidate]

    prompt = f"""
You are a QA reviewer for short-form video content.

Choose the best script based on:
- hook strength
- clarity
- pacing
- platform suitability
- audience fit

Candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

{{
  "best_index": 0,
  "reason": "...",
  "quality_score": 8.7
}}
"""

    response = generate_with_haiku(prompt)

    try:
        parsed = parse_json_response(response)
        best_index = parsed["best_index"]
        reason = parsed["reason"]
        quality_score = parsed["quality_score"]
    except Exception:
        best_index = 0
        reason = "Fallback selection due to parsing failure."
        quality_score = 7.5

    best_script = candidates[best_index] if candidates else {}

    return {
        "script_candidates": candidates,
        "best_script": best_script,
        "qa_output": {
            "selected_script": best_index,
            "reason": reason,
            "quality_score": quality_score,
        },
    }