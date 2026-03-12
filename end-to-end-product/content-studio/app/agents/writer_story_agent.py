"""
Story writer agent.

Creates a more narrative short-form script with a storytelling angle.
"""

import json

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
from app.services.json_utils import parse_json_response
from app.services.language_utils import get_language_instruction


def writer_story_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a story-driven short-form script candidate.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the story writer's script candidate.
    """

    print("RUNNING writer_story")
    
    request = state["request"]
    topic = request["topic"]

    language = state["request"].get("language", "en")
    language_instruction = get_language_instruction(language)

    prompt = f"""
You are a short-form video writer specializing in story-driven scripts.

Write one script for:
Topic: {topic}
Platform: {request["platform"]}
Audience: {request["audience"]}
Tone: {request["tone"]}
Duration: {request["duration_sec"]} seconds

{language_instruction}

Style requirements:
- narrative opening
- clear setup -> insight -> takeaway structure
- emotionally engaging but concise
- clear CTA
- do not invent statistics, company actions, or unsupported factual claims

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

{{
  "style": "story",
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
            "style": "story",
            "hook": f"It started quietly, and now {topic} is everywhere.",
            "script": f"What seemed niche about {topic} has now become part of a much bigger cultural shift.",
            "cta": "Do you think this trend lasts?",
        }

    return {
        "writer_story_output": parsed,
    }