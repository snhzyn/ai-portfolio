"""
Storyboard agent.

Generates a production-ready storyboard package for short-form video content
based on the selected script and requested output language.
"""

import re

from app.schemas.state import ContentStudioState
from app.services.korean_text_utils import object_particle, polish_topic_for_visual


def _build_short_label(text: str, fallback: str, max_words: int = 4) -> str:
    """
    Build a short on-screen label from a text fragment.

    Args:
        text: Source text to shorten.
        fallback: Fallback label if the source text is empty.
        max_words: Maximum number of words to keep.

    Returns:
        A shortened label string.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return fallback

    words = cleaned.split()
    short = " ".join(words[:max_words]).upper()
    return short


def _extract_short_voiceover(script_text: str, fallback: str, max_sentences: int = 2) -> str:
    """
    Extract a short voiceover summary from a longer script.

    Args:
        script_text: Full script text.
        fallback: Fallback line if the script text is empty.
        max_sentences: Maximum number of sentences to keep.

    Returns:
        A shorter, more natural voiceover line.
    """
    cleaned = " ".join(script_text.strip().split())
    if not cleaned:
        return fallback

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return fallback

    return " ".join(sentences[:max_sentences]).strip()


def storyboard_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a storyboard and shot list for the video package.

    Priority of source material:
    1. final_topic_suggestion
    2. normalized_topic
    3. raw topic

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the storyboard_output field.
    """
    request = state["request"]
    language = request.get("language", "en")
    raw_topic = request["topic"]

    director_brief = state.get("director_brief") or {}
    normalized_topic = director_brief.get("normalized_topic", raw_topic)
    final_topic = state.get("final_topic_suggestion") or normalized_topic or raw_topic
    visual_topic = polish_topic_for_visual(final_topic)

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    script_text = script_source.get("script", "")
    cta = script_source.get("cta", "")

    short_summary = _extract_short_voiceover(
        script_text,
        fallback="핵심 내용을 짧고 명확하게 전달합니다." if language == "ko" else "Deliver the key point clearly and quickly.",
        max_sentences=2,
    )

    if language == "ko":
        scenes = [
            {
                "scene": 1,
                "time_range": "0-3s",
                "visual": f"{object_particle(visual_topic)} 상징적으로 보여주는 강한 오프닝 컷",
                "voiceover": hook or f"요즘 왜 다들 {visual_topic}에 주목할까요?",
                "on_screen_text": " ".join((hook or final_topic).split()[:4]),
            },
            {
                "scene": 2,
                "time_range": "3-8s",
                "visual": "관련 이미지, 화면 자료, 상징 장면을 빠르게 보여주는 몽타주",
                "voiceover": f"{visual_topic}이 왜 주목받는지 흐름부터 빠르게 짚어보겠습니다.",
                "on_screen_text": "핵심 흐름 정리",
            },
            {
                "scene": 3,
                "time_range": "8-18s",
                "visual": "핵심 포인트 2~3개를 자막과 함께 빠르게 제시",
                "voiceover": short_summary,
                "on_screen_text": "핵심 포인트",
            },
            {
                "scene": 4,
                "time_range": "18-26s",
                "visual": "의미를 확장해 보여주는 자료 화면 또는 반응 컷",
                "voiceover": f"결국 {visual_topic}을 이해하려면 배경과 맥락을 함께 보는 게 중요합니다.",
                "on_screen_text": "왜 중요할까?",
            },
            {
                "scene": 5,
                "time_range": "26-30s",
                "visual": "댓글 유도용 엔드카드 또는 반응 컷",
                "voiceover": cta or "여러분은 어떻게 생각하시나요? 댓글로 알려주세요.",
                "on_screen_text": "당신의 생각은?",
            },
        ]

        output = {
            "agent_name": "storyboard",
            "summary": "숏폼 영상용 스토리보드 및 샷 리스트 생성",
            "scenes": scenes,
            "editing_style": "빠른 점프컷, 큰 중앙 자막, 첫 3초 강한 훅 중심 구성",
        }
    else:
        scenes = [
            {
                "scene": 1,
                "time_range": "0-3s",
                "visual": f"Strong opening visual related to {visual_topic}",
                "voiceover": hook or f"Why is everyone suddenly talking about {visual_topic}?",
                "on_screen_text": _build_short_label(hook, "WHY NOW?", max_words=4),
            },
            {
                "scene": 2,
                "time_range": "3-8s",
                "visual": "Fast montage of related visuals, references, and contextual cutaways",
                "voiceover": f"Let’s quickly break down why {visual_topic} is getting so much attention.",
                "on_screen_text": "QUICK BREAKDOWN",
            },
            {
                "scene": 3,
                "time_range": "8-18s",
                "visual": "Quick visual breakdown of 2-3 key points with bold captions",
                "voiceover": short_summary,
                "on_screen_text": "KEY POINTS",
            },
            {
                "scene": 4,
                "time_range": "18-26s",
                "visual": "Wider context visuals, reaction shots, or supporting footage",
                "voiceover": f"To really understand {visual_topic}, you need both the background and the bigger picture.",
                "on_screen_text": "WHY IT MATTERS",
            },
            {
                "scene": 5,
                "time_range": "26-30s",
                "visual": "End card or reaction frame optimized for comments",
                "voiceover": cta or "What do you think? Let me know in the comments.",
                "on_screen_text": "YOUR TAKE?",
            },
        ]

        output = {
            "agent_name": "storyboard",
            "summary": "Generated storyboard and shot list for short-form video",
            "scenes": scenes,
            "editing_style": "fast cuts, large captions, high-retention pacing, strong first-3-second hook",
        }

    return {
        "storyboard_output": output,
    }