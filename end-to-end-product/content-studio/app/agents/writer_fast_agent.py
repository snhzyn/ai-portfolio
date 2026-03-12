"""
Fast writer agent.

Creates a punchy, high-speed short-form script optimized for fast retention.
"""

import json

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
from app.services.json_utils import parse_json_response
from app.services.language_utils import get_language_instruction


def writer_fast_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a fast-paced short-form script candidate.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the fast writer's script candidate.
    """

    print("RUNNING writer_fast")

    request = state["request"]
    topic = request["topic"]

    language = state["request"].get("language", "en")
    language_instruction = get_language_instruction(language)

    prompt = f"""
You are a short-form video writer specializing in fast, punchy, high-retention scripts.

Write one script for:
Topic: {topic}
Platform: {request["platform"]}
Audience: {request["audience"]}
Tone: {request["tone"]}
Duration: {request["duration_sec"]} seconds

{language_instruction}

Style requirements:
- ultra-strong opening hook
- short sentences
- fast pacing
- highly scannable delivery
- clear CTA
- do not invent statistics, company actions, or unsupported factual claims

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

{{
  "style": "fast",
  "hook": "...",
  "script": "...",
  "cta": "..."
}}
"""

    response = generate_with_sonnet(prompt)

    try:
        parsed = parse_json_response(response)
    except Exception:
        parsed = {
            "style": "fast",
            "hook": f"Why is {topic} suddenly everywhere?",
            "script": f"{topic} is blowing up fast. Here's the reason people can't stop talking about it.",
            "cta": "What do you think?",
        }

    return {
        "writer_fast_output": parsed,
    }