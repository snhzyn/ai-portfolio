"""
Research agent.

Builds a lightweight topic research package to support downstream writing agents.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_haiku


def research_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a lightweight research package for the requested topic.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing research_output.
    """
    request = state["request"]
    director_brief = state.get("director_brief") or {}

    topic = request["topic"]
    audience = request.get("audience", "general audience")
    tone = request.get("tone", "engaging")
    reference_text = request.get("reference_text", "")
    language = request.get("language", "en")

    normalized_topic = director_brief.get("normalized_topic", topic)
    core_angle = director_brief.get("core_angle", topic)
    audience_frame = director_brief.get("audience_frame", audience)
    content_goal = director_brief.get("content_goal", "")

    prompt = f"""
You are a Research Agent for a short-form AI content studio.

Your task is to create a lightweight research package that helps script-writing agents.
Do not write the final script. Instead, produce structured support material.

Request:
- Topic: {topic}
- Normalized topic: {normalized_topic}
- Core angle: {core_angle}
- Audience: {audience}
- Audience frame: {audience_frame}
- Tone: {tone}
- Language: {language}
- Reference text: {reference_text}
- Content goal: {content_goal}

Return valid JSON only.
Do not include markdown fences.

Output schema:
{{
  "background_points": ["3-5 short points that explain the context"],
  "key_claims": ["3-5 concise claims or takeaways relevant for the script"],
  "supporting_examples": ["2-4 examples, metaphors, or concrete references"],
  "risk_or_uncertainty": ["Optional factual caution notes if needed"],
  "recommended_focus": "What the writers should emphasize most"
}}
"""

    try:
        response = generate_with_haiku(prompt)
        parsed = parse_json_response(response)
    except Exception:
        parsed = {
            "background_points": [normalized_topic],
            "key_claims": [core_angle],
            "supporting_examples": [],
            "risk_or_uncertainty": [],
            "recommended_focus": core_angle,
        }

    return {
        "research_output": {
            "agent_name": "research",
            "summary": "Lightweight topic research package",
            "background_points": parsed.get("background_points", []),
            "key_claims": parsed.get("key_claims", []),
            "supporting_examples": parsed.get("supporting_examples", []),
            "risk_or_uncertainty": parsed.get("risk_or_uncertainty", []),
            "recommended_focus": parsed.get("recommended_focus", core_angle),
        }
    }