"""
Fast writer agent.

Generates a concise, high-retention short-form script optimized for speed,
clarity, and direct delivery.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_sonnet


def writer_fast_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a fast-paced short-form script candidate.
    """
    request = state["request"]
    director_brief = state.get("director_brief") or {}
    research_output = state.get("research_output") or {}

    topic = request["topic"]
    platform = request["platform"]
    audience = request["audience"]
    tone = request["tone"]
    duration_sec = request["duration_sec"]
    language = request.get("language", "en")
    reference_text = request.get("reference_text", "")

    normalized_topic = director_brief.get("normalized_topic", topic)
    core_angle = director_brief.get("core_angle", topic)
    audience_frame = director_brief.get("audience_frame", audience)
    content_goal = director_brief.get("content_goal", "Generate a strong short-form content package")

    background_points = research_output.get("background_points", [])
    key_claims = research_output.get("key_claims", [])
    supporting_examples = research_output.get("supporting_examples", [])
    recommended_focus = research_output.get("recommended_focus", core_angle)

    if duration_sec == 15:
        duration_instruction = (
            "Keep it extremely tight. Use a strong hook, one core point, and a short CTA. "
            "Avoid extra explanation."
        )
    elif duration_sec == 30:
        duration_instruction = (
            "Keep it concise and punchy. Use a strong hook, 2-3 quick points, and a clear CTA."
        )
    elif duration_sec == 45:
        duration_instruction = (
            "Allow slightly more explanation, but keep momentum high. "
            "Use a hook, several clear points, and a CTA."
        )
    else:
        duration_instruction = (
            "You have more room for explanation, but it should still feel like short-form video. "
            "Use a strong hook, structured points, and a strong CTA."
        )

    prompt = f"""
You are the FAST writer agent for an AI short-form content studio.

Your job is to write a concise, high-retention short-form script optimized for speed, clarity, and direct delivery.

Request:
- Topic: {topic}
- Normalized topic: {normalized_topic}
- Core angle: {core_angle}
- Audience: {audience}
- Audience frame: {audience_frame}
- Platform: {platform}
- Tone: {tone}
- Duration: {duration_sec} seconds
- Language: {language}
- Reference text: {reference_text}
- Content goal: {content_goal}

Research support:
- Background points: {background_points}
- Key claims: {key_claims}
- Supporting examples: {supporting_examples}
- Recommended focus: {recommended_focus}

Writing instructions:
- Style: fast
- Prioritize speed, clarity, and punch
- Make the script easy to edit into short-form video
- {duration_instruction}

Return valid JSON only.
Do not include markdown fences.
Do not include commentary before or after the JSON.

Output schema:
{{
  "style": "fast",
  "hook": "...",
  "script": "...",
  "cta": "..."
}}
"""

    try:
        response = generate_with_sonnet(prompt)
        parsed = parse_json_response(response)
    except Exception:
        parsed = {
            "style": "fast",
            "hook": normalized_topic,
            "script": f"{normalized_topic}에 대해 핵심만 빠르게 정리합니다." if language == "ko" else f"A fast breakdown of {normalized_topic}.",
            "cta": "더 궁금하면 댓글로 남겨주세요." if language == "ko" else "Let me know what you think in the comments.",
        }

    candidate = {
        "style": parsed.get("style", "fast"),
        "hook": parsed.get("hook", ""),
        "script": parsed.get("script", ""),
        "cta": parsed.get("cta", ""),
    }

    return {
        "writer_fast_output": candidate,
    }