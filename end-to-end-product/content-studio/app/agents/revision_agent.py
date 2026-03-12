"""
Revision agent.

Improves the selected script using QA feedback and suggests
a final packaging-friendly topic.
"""

import json

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
from app.services.json_utils import parse_json_response
from app.services.language_utils import get_language_instruction


def revision_node(state: ContentStudioState) -> ContentStudioState:
    """
    Improve the selected script based on QA feedback and suggest
    a cleaner final topic for packaging.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the revised script and final topic suggestion.
    """
    request = state["request"]
    topic = request["topic"]
    language = request.get("language", "en")
    language_instruction = get_language_instruction(language)

    director_brief = state.get("director_brief") or {}
    normalized_topic = director_brief.get("normalized_topic", topic)

    script = state.get("best_script", {})

    prompt = f"""
You are improving a short-form video script.

Original topic:
{topic}

Current normalized topic:
{normalized_topic}

Original script:
{json.dumps(script, ensure_ascii=False, indent=2)}

{language_instruction}

Tasks:
1. Improve the script by:
- making the hook stronger
- tightening pacing
- improving platform-native phrasing
- keeping the same core idea
- making the CTA more natural

2. Also suggest a short, natural, production-friendly final topic
for titles, thumbnails, editor briefs, and video generation prompts.

Constraints:
- keep the tone witty and short-form-native, but brand-safe
- avoid unnecessarily aggressive, insulting, or overly niche slang
- avoid defamatory, misleading, or overly absolute claims
- do not invent unsupported factual claims
- avoid making claims about companies, statistics, or events unless already provided
- keep the output suitable for a broad public audience
- the final topic should be concise, clear, and natural
- the final topic should work well as a packaging label, not as a question

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

{{
  "style": "...",
  "hook": "...",
  "script": "...",
  "cta": "...",
  "final_topic_suggestion": "..."
}}
"""

    response = generate_with_sonnet(prompt)

    try:
        revised = parse_json_response(response)
    except Exception:
        revised = {
            "style": script.get("style", ""),
            "hook": script.get("hook", ""),
            "script": script.get("script", ""),
            "cta": script.get("cta", ""),
            "final_topic_suggestion": normalized_topic,
        }

    revised_script = {
        "style": revised.get("style", script.get("style", "")),
        "hook": revised.get("hook", script.get("hook", "")),
        "script": revised.get("script", script.get("script", "")),
        "cta": revised.get("cta", script.get("cta", "")),
    }

    final_topic_suggestion = revised.get("final_topic_suggestion", normalized_topic)

    return {
        "revised_script": revised_script,
        "final_topic_suggestion": final_topic_suggestion,
    }