"""
Story writer agent.

Generates a narrative-driven short-form script optimized for emotional flow,
clarity, and audience connection.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_sonnet


def writer_story_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a story-style short-form script candidate.
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
            "Even though this is story-style, keep it very short. "
            "Use a story-flavored hook and one quick narrative turn."
        )
    elif duration_sec == 30:
        duration_instruction = (
            "Keep the narrative compact. Use a clear setup, insight, and CTA."
        )
    elif duration_sec == 45:
        duration_instruction = (
            "Allow a slightly fuller narrative arc while keeping short-form pacing."
        )
    else:
        duration_instruction = (
            "Use a fuller narrative arc, but keep the delivery social-media-native and easy to follow."
        )

    prompt = f"""
You are the STORY writer agent for an AI short-form content studio.

Your job is to write a narrative-driven short-form script that feels emotionally coherent,
human, and easy to follow.

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
- Style: story
- Make the script feel like a compelling short-form narrative
- Prioritize emotional flow, coherence, and relatability
- {duration_instruction}

Return valid JSON only.
Do not include markdown fences.
Do not include commentary before or after the JSON.

Output schema:
{{
  "style": "story",
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
            "style": "story",
            "hook": normalized_topic,
            "script": f"{normalized_topic}에 대한 짧은 이야기형 설명입니다." if language == "ko" else f"A short narrative-style explanation of {normalized_topic}.",
            "cta": "공감되면 댓글로 알려주세요." if language == "ko" else "Let me know if this resonates with you.",
        }

    candidate = {
        "style": parsed.get("style", "story"),
        "hook": parsed.get("hook", ""),
        "script": parsed.get("script", ""),
        "cta": parsed.get("cta", ""),
    }

    return {
        "writer_story_output": candidate,
    }