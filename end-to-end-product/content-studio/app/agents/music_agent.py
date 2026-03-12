"""
Music agent.

Generates BGM direction, a Suno-ready music prompt, and editing notes
based on the selected script, normalized topic, tone, and requested output language.
"""

from app.schemas.state import ContentStudioState


def music_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate music direction and editing notes for the short-form video.

    This agent does not create music directly. Instead, it creates a
    production-ready music prompt that can be used in tools such as Suno,
    along with editing notes for short-form video workflows.

    Priority of source material:
    1. revised_script
    2. best_script
    3. normalized_topic
    4. raw topic fallback

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the music_output field.
    """
    request = state["request"]
    language = request.get("language", "en")
    raw_topic = request["topic"]
    tone = request["tone"]

    director_brief = state.get("director_brief") or {}
    normalized_topic = director_brief.get("normalized_topic", raw_topic)

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    script_text = script_source.get("script", "")
    script_mood = script_text[:220] if script_text else "short-form, trendy, visually engaging social content"

    if language == "ko":
        bgm_direction = "경쾌하고 세련되며 숏폼 편집에 잘 맞는 현대적인 배경음악"

        suno_prompt = (
            f"{normalized_topic}를 주제로 한 30초 숏폼 영상용 배경음악을 만들어줘. "
            f"톤은 '{tone}'이고, 전체 분위기는 트렌디하고 세련되며 에너지가 있어야 해. "
            f"과하게 웅장하지 말고, 짧은 영상 편집에 잘 맞도록 리듬감 있고 임팩트 있게 구성해줘. "
            f"보컬 없이, 자막과 내레이션 편집에 잘 어울리게 만들어줘. "
            f"오프닝 훅의 분위기는 '{hook}'이고, 영상 스크립트의 무드는 다음과 같아: {script_mood}"
        )

        editing_notes = [
            "첫 3초에 강한 비트 포인트 배치",
            "장면 전환마다 리듬감 있는 컷 편집",
            "큰 중앙 자막과 빠른 점프컷 사용",
            "2~3초마다 시각적 변화 주기",
            "엔드카드 직전에 볼륨을 살짝 낮춰 CTA를 강조",
        ]

        output = {
            "agent_name": "music",
            "summary": "BGM 방향성과 Suno 프롬프트 생성",
            "bgm_direction": bgm_direction,
            "suno_prompt": suno_prompt,
            "editing_notes": editing_notes,
            "hook_reference": hook,
        }
    else:
        bgm_direction = "upbeat, modern, stylish, and optimized for short-form retention"

        suno_prompt = (
            f"Create a 30-second background track for a short-form video about {normalized_topic}. "
            f"Tone: {tone}. "
            f"The track should feel modern, energetic, stylish, and highly editable for social video. "
            f"Avoid overly cinematic arrangements. Keep it clean, punchy, rhythmic, and suitable for fast cuts. "
            f"No vocals. Make it easy to edit around captions and voiceover. "
            f"The opening hook mood is '{hook}'. "
            f"The script mood is: {script_mood}"
        )

        editing_notes = [
            "Hit a strong beat within the first 3 seconds",
            "Use quick cuts synced to rhythm changes",
            "Add large center captions for key hook moments",
            "Keep visual changes every 2-3 seconds",
            "Lower the BGM slightly before the CTA for clarity",
        ]

        output = {
            "agent_name": "music",
            "summary": "Generated BGM direction, Suno prompt, and editing notes",
            "bgm_direction": bgm_direction,
            "suno_prompt": suno_prompt,
            "editing_notes": editing_notes,
            "hook_reference": hook,
        }

    return {
        "music_output": output,
    }