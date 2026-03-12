"""
Viral writer agent.

Creates a more provocative and curiosity-driven short-form script.
"""

import json

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
from app.services.json_utils import parse_json_response
from app.services.language_utils import get_language_instruction


def writer_viral_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a viral-style short-form script candidate.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the viral writer's script candidate.
    """

    print("RUNNING writer_viral")

    request = state["request"]
    topic = request["topic"]

    language = state["request"].get("language", "en")
    language_instruction = get_language_instruction(language)

    prompt = f"""
You are a short-form video writer specializing in viral, curiosity-driven scripts.

Write one script for:
Topic: {topic}
Platform: {request["platform"]}
Audience: {request["audience"]}
Tone: {request["tone"]}
Duration: {request["duration_sec"]} seconds

{language_instruction}

Style requirements:
- curiosity-first hook
- bold framing without sounding fake
- high shareability
- strong CTA
- do not invent statistics, company actions, or unsupported factual claims

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

{{
  "style": "viral",
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
            "style": "viral",
            "hook": f"The real reason {topic} is trending isn't what you think.",
            "script": f"Everyone thinks {topic} is just hype, but there's a deeper reason it's spreading so fast.",
            "cta": "Trend or overhyped?",
        }

    return {
        "writer_viral_output": parsed,
    }