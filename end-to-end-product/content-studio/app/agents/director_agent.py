"""
Director agent.

Interprets the user request, defines the creative angle,
and decides which specialist agents should run.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_haiku


_REQUIRED_AGENTS = [
    "writer_fast",
    "writer_story",
    "writer_viral",
    "storyboard",
    "title_thumbnail",
    "music",
]


def _normalize_agent_plan(agent_plan: list[str] | None) -> list[str]:
    """
    Normalize agent plan and ensure required downstream agents are present.
    """
    if not agent_plan:
        return _REQUIRED_AGENTS.copy()

    cleaned = []
    seen = set()

    for agent in agent_plan:
        if agent in _REQUIRED_AGENTS and agent not in seen:
            cleaned.append(agent)
            seen.add(agent)

    for agent in _REQUIRED_AGENTS:
        if agent not in seen:
            cleaned.append(agent)

    return cleaned


def director_node(state: ContentStudioState) -> ContentStudioState:
    """
    Analyze the user request and generate a structured creative brief.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state including the director brief and planned agents.
    """
    request = state["request"]

    topic = request["topic"]
    platform = request["platform"]
    audience = request["audience"]
    tone = request["tone"]
    duration_sec = request["duration_sec"]
    reference_text = request.get("reference_text", "")
    language = request.get("language", "en")

    prompt = f"""
You are the Director of an AI Content Production Studio.

Your job is to interpret the request and create a creative brief for downstream agents.

User request:
- Topic: {topic}
- Platform: {platform}
- Audience: {audience}
- Tone: {tone}
- Duration: {duration_sec} seconds
- Language: {language}
- Reference text: {reference_text}

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

Output schema:
{{
  "core_angle": "A short, clear central angle for the content",
  "normalized_topic": "A cleaner and more production-friendly topic label",
  "audience_frame": "How the topic should be framed for this audience",
  "content_goal": "What the content should help the viewer understand or feel",
  "risk_notes": ["Optional notes about ambiguity, sensitivity, or factual caution"],
  "agent_plan": ["writer_fast", "writer_story", "writer_viral", "storyboard", "title_thumbnail", "music"]
}}
"""

    try:
        response = generate_with_haiku(prompt)
        parsed = parse_json_response(response)
    except Exception:
        parsed = {
            "core_angle": topic,
            "normalized_topic": topic,
            "audience_frame": audience,
            "content_goal": "Generate a strong short-form video package",
            "risk_notes": [],
            "agent_plan": _REQUIRED_AGENTS.copy(),
        }

    planned_agents = _normalize_agent_plan(parsed.get("agent_plan"))

    return {
        "director_brief": {
            "core_angle": parsed.get("core_angle", topic),
            "normalized_topic": parsed.get("normalized_topic", topic),
            "audience_frame": parsed.get("audience_frame", audience),
            "content_goal": parsed.get("content_goal", "Generate a strong short-form video package"),
            "risk_notes": parsed.get("risk_notes", []),
            "language": language,
        },
        "planned_agents": planned_agents,
    }