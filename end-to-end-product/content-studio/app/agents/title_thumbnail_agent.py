"""
Title and thumbnail agent.

Generates title options, thumbnail text, caption, and hashtags
for the final video package based on the selected script and language.
"""

from app.schemas.state import ContentStudioState


def _short_thumbnail_from_hook(hook: str, fallback: str) -> str:
    """
    Build short thumbnail text from a hook.

    Args:
        hook: Hook text from the selected script.
        fallback: Fallback text if the hook is empty.

    Returns:
        A short thumbnail-style text string.
    """
    if not hook:
        return fallback

    cleaned = hook.strip().replace("?", "").replace("!", "")
    words = cleaned.split()

    if len(words) <= 3:
        return cleaned.upper()

    return " ".join(words[:3]).upper()


def title_thumbnail_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate publish-ready metadata for the short-form video.

    This agent creates titles, thumbnail copy, captions, and hashtags
    using the current creative brief and the strongest available script.

    Priority of source material:
    1. revised_script
    2. best_script
    3. normalized_topic
    4. raw topic fallback

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the title_thumbnail_output field.
    """
    request = state["request"]
    language = request.get("language", "en")
    raw_topic = request["topic"]

    director_brief = state.get("director_brief") or {}
    core_angle = director_brief.get("core_angle", raw_topic)
    normalized_topic = director_brief.get("normalized_topic", raw_topic)

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    cta = script_source.get("cta", "")

    if language == "ko":
        titles = [
            f"요즘 왜 {normalized_topic}가 주목받을까",
            f"{normalized_topic} 열풍의 진짜 이유",
            f"{normalized_topic}, 단순 유행이 아닌 이유",
            f"{normalized_topic} 한 번에 이해하기",
            f"왜 다들 {normalized_topic}를 찾을까",
        ]

        thumbnail_text = [
            _short_thumbnail_from_hook(hook, "왜 뜰까?"),
            "진짜 이유",
            "단순 유행 아님",
        ]

        caption = (
            f"{normalized_topic}가 왜 갑자기 주목받고 있는지 핵심만 짧고 빠르게 정리했습니다. "
            f"{cta or '여러분의 생각도 댓글로 남겨주세요.'}"
        )

        hashtags = [
            "#숏폼",
            "#트렌드",
            "#콘텐츠",
            "#바이럴",
            "#말차" if "말차" in normalized_topic else "#이슈",
        ]

        output = {
            "agent_name": "title_thumbnail",
            "summary": "제목, 썸네일, 캡션, 해시태그 패키지 생성",
            "titles": titles,
            "thumbnail_text": thumbnail_text,
            "caption": caption,
            "hashtags": hashtags,
            "hook_reference": hook,
            "angle_reference": core_angle,
        }
    else:
        titles = [
            f"Why {normalized_topic.capitalize()} Is Suddenly Everywhere",
            f"The Real Reason {normalized_topic.capitalize()} Is Trending",
            f"What’s Driving the {normalized_topic.capitalize()} Hype?",
            f"{normalized_topic.capitalize()} Explained in 30 Seconds",
            f"Why Everyone Is Talking About {normalized_topic.capitalize()} Right Now",
        ]

        thumbnail_text = [
            _short_thumbnail_from_hook(hook, "WHY MATCHA?"),
            "MORE THAN HYPE",
            "MATCHA TAKEOVER" if "matcha" in normalized_topic.lower() else "TREND ALERT",
        ]

        caption = (
            f"{normalized_topic.capitalize()} is getting a lot of attention right now, and there are real reasons behind it. "
            f"{cta or 'What do you think? Let me know in the comments.'}"
        )

        hashtags = [
            "#trend",
            "#shorts",
            "#viralcontent",
            "#contentstudio",
            "#matcha" if "matcha" in normalized_topic.lower() else "#topic",
        ]

        output = {
            "agent_name": "title_thumbnail",
            "summary": "Generated title, thumbnail, caption, and hashtag package",
            "titles": titles,
            "thumbnail_text": thumbnail_text,
            "caption": caption,
            "hashtags": hashtags,
            "hook_reference": hook,
            "angle_reference": core_angle,
        }

    return {
        "title_thumbnail_output": output,
    }