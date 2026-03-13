"""
Storyboard agent.

Generates a production-ready storyboard package for short-form video content
based on the selected script, requested duration, and output language.
"""

import re

from app.schemas.state import ContentStudioState
from app.services.korean_text_utils import object_particle, polish_topic_for_visual


def _build_short_label(text: str, fallback: str, max_words: int = 4) -> str:
    """
    Build a short on-screen label from a text fragment.
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
    """
    cleaned = " ".join(script_text.strip().split())
    if not cleaned:
        return fallback

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return fallback

    return " ".join(sentences[:max_sentences]).strip()


def _get_scene_ranges(duration_sec: int) -> list[str]:
    """
    Return time ranges based on the requested short-form video duration.

    Args:
        duration_sec: Requested video duration in seconds.

    Returns:
        List of time ranges for storyboard scenes.
    """
    if duration_sec == 15:
        return ["0-2s", "2-5s", "5-11s", "11-15s"]

    if duration_sec == 30:
        return ["0-3s", "3-8s", "8-18s", "18-26s", "26-30s"]

    if duration_sec == 45:
        return ["0-3s", "3-8s", "8-18s", "18-28s", "28-38s", "38-45s"]

    if duration_sec == 60:
        return ["0-4s", "4-10s", "10-20s", "20-32s", "32-44s", "44-54s", "54-60s"]

    # fallback
    return ["0-3s", "3-8s", "8-18s", "18-26s", "26-30s"]


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
    duration_sec = request.get("duration_sec", 30)

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
        max_sentences=2 if duration_sec <= 30 else 3,
    )

    ranges = _get_scene_ranges(duration_sec)

    if language == "ko":
        if duration_sec == 15:
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"{object_particle(visual_topic)} 상징적으로 보여주는 강한 오프닝 컷",
                    "voiceover": hook or f"{visual_topic}, 왜 갑자기 뜨고 있을까요?",
                    "on_screen_text": " ".join((hook or final_topic).split()[:4]),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "관련 이미지와 핵심 장면을 빠르게 보여주는 몽타주",
                    "voiceover": f"{visual_topic}의 핵심만 빠르게 보겠습니다.",
                    "on_screen_text": "핵심만 요약",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "핵심 포인트를 자막과 함께 압축적으로 제시",
                    "voiceover": short_summary,
                    "on_screen_text": "핵심 포인트",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "엔드카드 또는 반응 컷",
                    "voiceover": cta or "여러분 생각도 댓글로 남겨주세요.",
                    "on_screen_text": "어떻게 생각해?",
                },
            ]

        elif duration_sec == 30:
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"{object_particle(visual_topic)} 상징적으로 보여주는 강한 오프닝 컷",
                    "voiceover": hook or f"요즘 왜 다들 {visual_topic}에 주목할까요?",
                    "on_screen_text": " ".join((hook or final_topic).split()[:4]),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "관련 이미지, 화면 자료, 상징 장면을 빠르게 보여주는 몽타주",
                    "voiceover": f"{visual_topic}이 왜 주목받는지 흐름부터 빠르게 짚어보겠습니다.",
                    "on_screen_text": "핵심 흐름 정리",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "핵심 포인트 2~3개를 자막과 함께 빠르게 제시",
                    "voiceover": short_summary,
                    "on_screen_text": "핵심 포인트",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "의미를 확장해 보여주는 자료 화면 또는 반응 컷",
                    "voiceover": f"결국 {visual_topic}을 이해하려면 배경과 맥락을 함께 보는 게 중요합니다.",
                    "on_screen_text": "왜 중요할까?",
                },
                {
                    "scene": 5,
                    "time_range": ranges[4],
                    "visual": "댓글 유도용 엔드카드 또는 반응 컷",
                    "voiceover": cta or "여러분은 어떻게 생각하시나요? 댓글로 알려주세요.",
                    "on_screen_text": "당신의 생각은?",
                },
            ]

        elif duration_sec == 45:
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"{object_particle(visual_topic)} 상징적으로 보여주는 강한 오프닝 컷",
                    "voiceover": hook or f"{visual_topic}, 왜 이렇게 주목받을까요?",
                    "on_screen_text": " ".join((hook or final_topic).split()[:4]),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "관련 자료와 트렌드 장면을 빠르게 보여주는 컷",
                    "voiceover": f"먼저 {visual_topic}의 배경부터 짧게 보겠습니다.",
                    "on_screen_text": "배경 먼저",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "핵심 포인트 1~2개 제시",
                    "voiceover": short_summary,
                    "on_screen_text": "핵심 1",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "추가 근거나 예시를 보여주는 장면",
                    "voiceover": f"{visual_topic}이 주목받는 데에는 여러 요소가 함께 작용합니다.",
                    "on_screen_text": "핵심 2",
                },
                {
                    "scene": 5,
                    "time_range": ranges[4],
                    "visual": "맥락과 의미를 넓혀 보여주는 반응 컷",
                    "voiceover": f"그래서 {visual_topic}은 단순한 유행이나 사건으로 보기 어렵습니다.",
                    "on_screen_text": "더 큰 맥락",
                },
                {
                    "scene": 6,
                    "time_range": ranges[5],
                    "visual": "엔드카드 또는 댓글 유도 컷",
                    "voiceover": cta or "여러분 생각도 댓글로 남겨주세요.",
                    "on_screen_text": "당신의 생각은?",
                },
            ]

        else:  # 60 sec
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"{object_particle(visual_topic)} 상징적으로 보여주는 강한 오프닝 컷",
                    "voiceover": hook or f"{visual_topic}, 왜 이렇게 화제가 될까요?",
                    "on_screen_text": " ".join((hook or final_topic).split()[:4]),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "관련 이미지, 자료, 상징 장면을 빠르게 보여주는 도입 컷",
                    "voiceover": f"먼저 {visual_topic}의 배경을 간단히 보겠습니다.",
                    "on_screen_text": "배경 정리",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "핵심 포인트 첫 번째 설명 장면",
                    "voiceover": short_summary,
                    "on_screen_text": "핵심 1",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "추가 포인트와 예시를 보여주는 장면",
                    "voiceover": f"{visual_topic}을 이해하려면 추가적인 맥락도 함께 봐야 합니다.",
                    "on_screen_text": "핵심 2",
                },
                {
                    "scene": 5,
                    "time_range": ranges[4],
                    "visual": "사회적 반응, 예시, 확장 장면",
                    "voiceover": f"이 흐름이 실제로 어떻게 이어지는지도 중요합니다.",
                    "on_screen_text": "확장 맥락",
                },
                {
                    "scene": 6,
                    "time_range": ranges[5],
                    "visual": "정리와 의미를 강조하는 장면",
                    "voiceover": f"결국 {visual_topic}은 여러 요소가 겹쳐 만들어진 결과입니다.",
                    "on_screen_text": "핵심 정리",
                },
                {
                    "scene": 7,
                    "time_range": ranges[6],
                    "visual": "엔드카드 또는 CTA 컷",
                    "voiceover": cta or "여러분 생각도 댓글로 남겨주세요.",
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
        if duration_sec == 15:
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"Strong opening visual related to {visual_topic}",
                    "voiceover": hook or f"Why is everyone suddenly talking about {visual_topic}?",
                    "on_screen_text": _build_short_label(hook, "WHY NOW?", max_words=4),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "Fast montage of related visuals and context",
                    "voiceover": f"Here’s the quick version of why {visual_topic} matters.",
                    "on_screen_text": "QUICK TAKE",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "Quick breakdown of the key point",
                    "voiceover": short_summary,
                    "on_screen_text": "KEY POINT",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "End card or reaction shot",
                    "voiceover": cta or "What do you think? Let me know in the comments.",
                    "on_screen_text": "YOUR TAKE?",
                },
            ]

        elif duration_sec == 30:
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"Strong opening visual related to {visual_topic}",
                    "voiceover": hook or f"Why is everyone suddenly talking about {visual_topic}?",
                    "on_screen_text": _build_short_label(hook, "WHY NOW?", max_words=4),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "Fast montage of related visuals, references, and contextual cutaways",
                    "voiceover": f"Let’s quickly break down why {visual_topic} is getting so much attention.",
                    "on_screen_text": "QUICK BREAKDOWN",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "Quick visual breakdown of 2-3 key points with bold captions",
                    "voiceover": short_summary,
                    "on_screen_text": "KEY POINTS",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "Wider context visuals, reaction shots, or supporting footage",
                    "voiceover": f"To really understand {visual_topic}, you need both the background and the bigger picture.",
                    "on_screen_text": "WHY IT MATTERS",
                },
                {
                    "scene": 5,
                    "time_range": ranges[4],
                    "visual": "End card or reaction frame optimized for comments",
                    "voiceover": cta or "What do you think? Let me know in the comments.",
                    "on_screen_text": "YOUR TAKE?",
                },
            ]

        elif duration_sec == 45:
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"Strong opening visual related to {visual_topic}",
                    "voiceover": hook or f"Why is {visual_topic} suddenly getting attention?",
                    "on_screen_text": _build_short_label(hook, "WHY NOW?", max_words=4),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "Intro montage with trend/context visuals",
                    "voiceover": f"Let’s start with the background behind {visual_topic}.",
                    "on_screen_text": "BACKGROUND",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "Break down key point one",
                    "voiceover": short_summary,
                    "on_screen_text": "KEY POINT 1",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "Additional examples or supporting footage",
                    "voiceover": f"There’s more than one reason {visual_topic} is resonating right now.",
                    "on_screen_text": "KEY POINT 2",
                },
                {
                    "scene": 5,
                    "time_range": ranges[4],
                    "visual": "Context expansion or audience relevance shot",
                    "voiceover": f"This is why {visual_topic} connects to a bigger shift or pattern.",
                    "on_screen_text": "BIGGER SHIFT",
                },
                {
                    "scene": 6,
                    "time_range": ranges[5],
                    "visual": "End card or reaction frame",
                    "voiceover": cta or "What do you think? Let me know in the comments.",
                    "on_screen_text": "YOUR TAKE?",
                },
            ]

        else:  # 60 sec
            scenes = [
                {
                    "scene": 1,
                    "time_range": ranges[0],
                    "visual": f"Strong opening visual related to {visual_topic}",
                    "voiceover": hook or f"Why is {visual_topic} becoming such a big topic?",
                    "on_screen_text": _build_short_label(hook, "WHY NOW?", max_words=4),
                },
                {
                    "scene": 2,
                    "time_range": ranges[1],
                    "visual": "Fast intro montage with context visuals",
                    "voiceover": f"First, let’s look at the background behind {visual_topic}.",
                    "on_screen_text": "BACKGROUND",
                },
                {
                    "scene": 3,
                    "time_range": ranges[2],
                    "visual": "Break down key point one with captions",
                    "voiceover": short_summary,
                    "on_screen_text": "KEY POINT 1",
                },
                {
                    "scene": 4,
                    "time_range": ranges[3],
                    "visual": "Expand with supporting visuals or examples",
                    "voiceover": f"Now let’s add the second layer of context.",
                    "on_screen_text": "KEY POINT 2",
                },
                {
                    "scene": 5,
                    "time_range": ranges[4],
                    "visual": "Show audience relevance or broader impact",
                    "voiceover": f"This also matters because it reflects a broader shift.",
                    "on_screen_text": "WHY IT MATTERS",
                },
                {
                    "scene": 6,
                    "time_range": ranges[5],
                    "visual": "Summarize with strong closing visuals",
                    "voiceover": f"So when you look at {visual_topic}, the bigger picture matters just as much as the headline.",
                    "on_screen_text": "BIG PICTURE",
                },
                {
                    "scene": 7,
                    "time_range": ranges[6],
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