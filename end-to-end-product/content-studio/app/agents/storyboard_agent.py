"""
Storyboard agent.

Generates a production-ready storyboard package for short-form video content
based on the selected script and requested output language.
"""

from app.schemas.state import ContentStudioState


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


def storyboard_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a storyboard and shot list for the video package.

    This agent builds scene-by-scene guidance that can be used directly
    in short-form editing tools such as CapCut or Premiere.

    Priority of source material:
    1. revised_script
    2. best_script
    3. normalized_topic
    4. raw topic fallback

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

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    script_text = script_source.get("script", "")
    cta = script_source.get("cta", "")

    if language == "ko":
        scenes = [
            {
                "scene": 1,
                "time_range": "0-3s",
                "visual": f"{normalized_topic}를 상징적으로 보여주는 강한 오프닝 컷",
                "voiceover": hook or f"요즘 왜 다들 {normalized_topic}에 주목할까요?",
                "on_screen_text": _build_short_label(hook, f"{normalized_topic} 왜 뜰까?", max_words=4),
            },
            {
                "scene": 2,
                "time_range": "3-8s",
                "visual": "제품 클로즈업, SNS 화면, 카페 장면 등 빠른 몽타주",
                "voiceover": "단순한 유행처럼 보이지만, 그 안에는 분명한 이유가 있습니다.",
                "on_screen_text": "단순한 유행 아님",
            },
            {
                "scene": 3,
                "time_range": "8-18s",
                "visual": "핵심 이유 2~3가지를 자막과 함께 빠르게 제시",
                "voiceover": (
                    script_text[:180]
                    if script_text
                    else f"{normalized_topic}가 주목받는 이유는 기능성, 분위기, 그리고 공유하기 쉬운 이미지 때문입니다."
                ),
                "on_screen_text": "왜 지금일까?",
            },
            {
                "scene": 4,
                "time_range": "18-26s",
                "visual": "트렌드를 상징하는 라이프스타일 컷과 반응형 장면",
                "voiceover": "이건 단순한 제품이 아니라 하나의 라이프스타일 신호가 되고 있습니다.",
                "on_screen_text": "라이프스타일 신호",
            },
            {
                "scene": 5,
                "time_range": "26-30s",
                "visual": "댓글 유도용 엔드카드 또는 반응 컷",
                "voiceover": cta or "여러분은 어떻게 생각하시나요? 댓글로 알려주세요.",
                "on_screen_text": "당신의 의견은?",
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
                "visual": f"Strong opening visual related to {normalized_topic}",
                "voiceover": hook or f"Why is everyone suddenly talking about {normalized_topic}?",
                "on_screen_text": _build_short_label(hook, "WHY NOW?", max_words=4),
            },
            {
                "scene": 2,
                "time_range": "3-8s",
                "visual": "Fast montage of trend signals, social media clips, and product close-ups",
                "voiceover": "It looks like a trend, but there’s a deeper reason this is getting so much attention.",
                "on_screen_text": "MORE THAN HYPE",
            },
            {
                "scene": 3,
                "time_range": "8-18s",
                "visual": "Quick visual breakdown of 2-3 key reasons with bold captions",
                "voiceover": (
                    script_text[:180]
                    if script_text
                    else f"{normalized_topic} is taking off because it combines function, identity, and shareability."
                ),
                "on_screen_text": "WHY NOW?",
            },
            {
                "scene": 4,
                "time_range": "18-26s",
                "visual": "Lifestyle cutaways showing cultural relevance and social signaling",
                "voiceover": "This is no longer just a product or habit. It is becoming part of a lifestyle signal.",
                "on_screen_text": "CULTURAL SIGNAL",
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