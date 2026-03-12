"""
Revision agent.

Improves the selected script using QA feedback.
"""

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
from app.services.json_utils import parse_json_response
from app.services.language_utils import get_language_instruction
import json


def revision_node(state: ContentStudioState) -> ContentStudioState:
    """
    Improve the selected script based on QA feedback.
    """

    script = state.get("best_script", {})
    topic = state["request"]["topic"]

    language = state["request"].get("language", "en")
    language_instruction = get_language_instruction(language)

    prompt = f"""
You are improving a short-form video script.

Topic: {topic}

Original script:
{json.dumps(script, ensure_ascii=False, indent=2)}

{language_instruction}

Improve the script by:
- making the hook stronger
- tightening pacing
- improving platform-native phrasing
- keeping the same core idea
- making the CTA more natural
- avoiding unnecessarily aggressive, insulting, or overly niche slang
- avoiding unsupported factual claims

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

{{
  "style": "...",
  "hook": "...",
  "script": "...",
  "cta": "..."
}}
"""

    response = generate_with_sonnet(prompt)

    try:
        revised = parse_json_response(response)
    except Exception:
        revised = script

    return {
        "revised_script": revised,
    }