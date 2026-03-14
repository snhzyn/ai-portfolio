"""
Title / Thumbnail agent.

Generates platform-ready titles, thumbnail text, caption, and hashtag package
based on the selected script, final topic, and requested output language.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_haiku
from app.services.korean_text_utils import polish_topic_for_visual


def _fallback_output(
    language: str,
    final_topic: str,
    hook: str,
    cta: str,
) -> dict:
    """
    Safe fallback output when LLM generation fails.
    """
    visual_topic = polish_topic_for_visual(final_topic)

    if language == "ko":
        titles = [
            f"{visual_topic}, 왜 주목받을까",
            f"{visual_topic} 핵심만 빠르게 정리",
            f"{visual_topic} 한 번에 이해하기",
            f"{visual_topic}, 왜 중요한가",
            f"{visual_topic}의 핵심 포인트",
        ]

        thumbnail_text = [
            hook[:18] if hook else visual_topic[:18],
            final_topic[:18],
            "핵심 이유 정리",
        ]

        caption = f"{visual_topic}을 짧고 빠르게 정리했습니다. {cta or '여러분의 생각도 댓글로 남겨주세요.'}"

        hashtags = [
            "#숏폼",
            "#트렌드",
            "#콘텐츠",
            "#바이럴",
            "#인사이트",
        ]

        return {
            "agent_name": "title_thumbnail",
            "summary": "제목, 썸네일, 캡션, 해시태그 패키지 생성",
            "titles": titles,
            "thumbnail_text": thumbnail_text,
            "caption": caption,
            "hashtags": hashtags,
            "hook_reference": hook,
            "final_topic_reference": final_topic,
        }

    titles = [
        f"Why {final_topic} Is Getting Attention",
        f"{final_topic} Explained Quickly",
        f"What You Should Know About {final_topic}",
        f"Why {final_topic} Matters",
        f"The Key Idea Behind {final_topic}",
    ]

    thumbnail_text = [
        hook[:20].upper() if hook else final_topic[:20].upper(),
        "WHY IT MATTERS",
        "QUICK BREAKDOWN",
    ]

    caption = f"A quick breakdown of {final_topic}. {cta or 'Let me know what you think in the comments.'}"

    hashtags = [
        "#shorts",
        "#trend",
        "#content",
        "#viral",
        "#insight",
    ]

    return {
        "agent_name": "title_thumbnail",
        "summary": "Generated title, thumbnail, caption, and hashtag package",
        "titles": titles,
        "thumbnail_text": thumbnail_text,
        "caption": caption,
        "hashtags": hashtags,
        "hook_reference": hook,
        "final_topic_reference": final_topic,
    }


def title_thumbnail_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a title / thumbnail / caption package for the selected script.
    """
    request = state["request"]
    language = request.get("language", "en")
    platform = request.get("platform", "youtube_shorts")
    raw_topic = request.get("topic", "")
    tone = request.get("tone", "")
    audience = request.get("audience", "")

    director_brief = state.get("director_brief") or {}
    normalized_topic = director_brief.get("normalized_topic", raw_topic)
    core_angle = director_brief.get("core_angle", raw_topic)
    final_topic = state.get("final_topic_suggestion") or normalized_topic or raw_topic

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    script_text = script_source.get("script", "")
    cta = script_source.get("cta", "")

    if language == "ko":
        prompt = f"""
You are the Title / Thumbnail Agent for an AI short-form content studio.

Your task is to generate:
1. 5 strong short-form video titles
2. 3 thumbnail text options
3. 1 caption
4. 5 hashtags

Important rules:
- Use the selected hook and revised script as the main source of truth.
- Do NOT generate generic explanatory titles.
- Titles should feel platform-native for {platform}.
- Prefer curiosity, emotional contrast, transformation, surprise, or strong framing.
- Thumbnail text must be short, visual, punchy, and readable.
- Avoid repeating the full topic as-is unless necessary.
- Caption should feel natural and social-media-friendly.
- Use Korean output only.

Context:
- Raw topic: {raw_topic}
- Final topic: {final_topic}
- Core angle: {core_angle}
- Platform: {platform}
- Audience: {audience}
- Tone: {tone}
- Hook: {hook}
- Revised script: {script_text}
- CTA: {cta}

Return valid JSON only.

Output schema:
{{
  "titles": ["...", "...", "...", "...", "..."],
  "thumbnail_text": ["...", "...", "..."],
  "caption": "...",
  "hashtags": ["#...", "#...", "#...", "#...", "#..."]
}}
"""
    else:
        prompt = f"""
You are the Title / Thumbnail Agent for an AI short-form content studio.

Your task is to generate:
1. 5 strong short-form video titles
2. 3 thumbnail text options
3. 1 caption
4. 5 hashtags

Important rules:
- Use the selected hook and revised script as the main source of truth.
- Do NOT generate generic explanatory titles.
- Titles should feel platform-native for {platform}.
- Prefer curiosity, emotional contrast, transformation, surprise, or strong framing.
- Thumbnail text must be short, visual, punchy, and readable.
- Avoid repeating the full topic as-is unless necessary.
- Caption should feel natural and social-media-friendly.
- Use English output only.

Context:
- Raw topic: {raw_topic}
- Final topic: {final_topic}
- Core angle: {core_angle}
- Platform: {platform}
- Audience: {audience}
- Tone: {tone}
- Hook: {hook}
- Revised script: {script_text}
- CTA: {cta}

Return valid JSON only.

Output schema:
{{
  "titles": ["...", "...", "...", "...", "..."],
  "thumbnail_text": ["...", "...", "..."],
  "caption": "...",
  "hashtags": ["#...", "#...", "#...", "#...", "#..."]
}}
"""

    try:
        response = generate_with_haiku(prompt)
        parsed = parse_json_response(response)

        titles = parsed.get("titles", [])
        thumbnail_text = parsed.get("thumbnail_text", [])
        caption = parsed.get("caption", "")
        hashtags = parsed.get("hashtags", [])

        if not isinstance(titles, list) or len(titles) < 3:
            raise ValueError("Invalid titles output")
        if not isinstance(thumbnail_text, list) or len(thumbnail_text) < 2:
            raise ValueError("Invalid thumbnail_text output")
        if not isinstance(hashtags, list) or len(hashtags) < 3:
            raise ValueError("Invalid hashtags output")

        output = {
            "agent_name": "title_thumbnail",
            "summary": "제목, 썸네일, 캡션, 해시태그 패키지 생성"
            if language == "ko"
            else "Generated title, thumbnail, caption, and hashtag package",
            "titles": titles[:5],
            "thumbnail_text": thumbnail_text[:3],
            "caption": caption,
            "hashtags": hashtags[:5],
            "hook_reference": hook,
            "angle_reference": core_angle,
            "final_topic_reference": final_topic,
        }

    except Exception:
        output = _fallback_output(
            language=language,
            final_topic=final_topic,
            hook=hook,
            cta=cta,
        )

    return {
        "title_thumbnail_output": output,
    }