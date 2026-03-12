"""
Title and thumbnail agent.

Generates title options, thumbnail text, caption, and hashtags
for the final video package based on the selected script and language.
"""

import re

from app.schemas.state import ContentStudioState
from app.services.korean_text_utils import object_particle, subject_particle, topic_particle


def _clean_phrase(text: str) -> str:
    """
    Clean a text phrase for reuse in titles or thumbnail text.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_short_thumbnail_phrases(hook: str, final_topic: str, language: str) -> list[str]:
    """
    Build short thumbnail-style phrases from the hook and topic.
    """
    hook_clean = _clean_phrase(hook).replace("?", "").replace("!", "")
    topic_clean = _clean_phrase(final_topic)

    if language == "ko":
        phrases = []

        if hook_clean:
            parts = re.split(r"[,.—:\-]", hook_clean)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                words = part.split()
                short = " ".join(words[:3]).strip()
                if short and short not in phrases:
                    phrases.append(short)

        generic_fallbacks = [
            topic_clean,
            "핵심 이유 정리",
            "왜 계속될까?",
        ]

        for fallback in generic_fallbacks:
            if fallback and fallback not in phrases:
                phrases.append(fallback)

        return phrases[:3]

    phrases = []

    if hook_clean:
        parts = re.split(r"[,.—:\-]", hook_clean)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            words = part.split()
            short = " ".join(words[:4]).strip().upper()
            if short and short not in phrases:
                phrases.append(short)

    if topic_clean:
        topic_words = topic_clean.split()
        short_topic = " ".join(topic_words[:4]).strip().upper()
        if short_topic and short_topic not in phrases:
            phrases.append(short_topic)

    generic_fallbacks = [
        "WHY NOW?",
        "MORE THAN HYPE",
        "TREND SHIFT",
    ]

    for fallback in generic_fallbacks:
        if fallback not in phrases:
            phrases.append(fallback)

    return phrases[:3]


def title_thumbnail_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate publish-ready metadata for the short-form video.
    """
    request = state["request"]
    language = request.get("language", "en")
    raw_topic = request["topic"]

    director_brief = state.get("director_brief") or {}
    core_angle = director_brief.get("core_angle", raw_topic)
    normalized_topic = director_brief.get("normalized_topic", raw_topic)
    final_topic = state.get("final_topic_suggestion") or normalized_topic or raw_topic

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    cta = script_source.get("cta", "")

    thumbnail_text = _extract_short_thumbnail_phrases(
        hook=hook,
        final_topic=final_topic,
        language=language,
    )

    if language == "ko":
        titles = [
            f"{final_topic}, 왜 주목받을까",
            f"{final_topic} 핵심만 빠르게 정리",
            f"{final_topic} 한 번에 이해하기",
            f"{final_topic}, 왜 중요한가",
            f"{final_topic}의 핵심 포인트",
        ]

        caption = (
            f"{object_particle(final_topic)} 짧고 빠르게 정리했습니다. "
            f"{cta or '여러분의 생각도 댓글로 남겨주세요.'}"
        )

        hashtags = [
            "#숏폼",
            "#트렌드",
            "#콘텐츠",
            "#바이럴",
            "#인사이트",
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
            "final_topic_reference": final_topic,
        }
    else:
        titles = [
            f"Why {final_topic.capitalize()} Matters Right Now",
            f"The Real Reason {final_topic.capitalize()} Is Trending",
            f"{final_topic.capitalize()} Explained in 30 Seconds",
            f"What’s Driving {final_topic.capitalize()}?",
            f"Why Everyone Is Talking About {final_topic.capitalize()}",
        ]

        caption = (
            f"{final_topic.capitalize()} is getting a lot of attention right now, and there are real reasons behind it. "
            f"{cta or 'What do you think? Let me know in the comments.'}"
        )

        hashtags = [
            "#trend",
            "#shorts",
            "#viralcontent",
            "#contentstudio",
            "#insight",
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
            "final_topic_reference": final_topic,
        }

    return {
        "title_thumbnail_output": output,
    }