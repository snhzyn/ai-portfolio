"""
Viral writer agent.

Generates a curiosity-driven, platform-native short-form script optimized for
engagement, retention, and shareability.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_sonnet


def writer_viral_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a viral-style short-form script candidate.
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
    content_goal = director_brief.get("content_goal", "Deliver a strong short-form content package")

    background_points = research_output.get("background_points", [])
    key_claims = research_output.get("key_claims", [])
    supporting_examples = research_output.get("supporting_examples", [])
    recommended_focus = research_output.get("recommended_focus", core_angle)

    if duration_sec == 15:
        duration_instruction = (
            "Make it highly compressed. Use a curiosity hook, one hard-hitting idea, and a quick CTA."
        )
    elif duration_sec == 30:
        duration_instruction = (
            "Use a strong curiosity hook, 2-3 memorable points, and a comment-friendly CTA."
        )
    elif duration_sec == 45:
        duration_instruction = (
            "Keep it energetic and shareable, but allow slightly more explanation."
        )
    else:
        duration_instruction = (
            "Maintain strong short-form momentum across a longer runtime with multiple strong beats."
        )

    prompt = f"""
You are the VIRAL writer agent for an AI short-form content studio.

Your job is to write a highly engaging, curiosity-driven, platform-native script
optimized for social retention and shareability.

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
- Style: viral
- Prioritize strong hooks, social-media-native rhythm, and shareability
- Make it feel native to short-form platforms
- {duration_instruction}

Return valid JSON only.
Do not include markdown fences.
Do not include commentary before or after the JSON.

Output schema:
{{
  "style": "viral",
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
            "style": "viral",
            "hook": normalized_topic,
            "script": f"{normalized_topic}의 핵심을 바이럴하게 정리합니다." if language == "ko" else f"A viral short-form breakdown of {normalized_topic}.",
            "cta": "여러분 생각은 어떤가요?" if language == "ko" else "What do you think?",
        }

    candidate = {
        "style": parsed.get("style", "viral"),
        "hook": parsed.get("hook", ""),
        "script": parsed.get("script", ""),
        "cta": parsed.get("cta", ""),
    }

    return {
        "writer_viral_output": candidate,
    }