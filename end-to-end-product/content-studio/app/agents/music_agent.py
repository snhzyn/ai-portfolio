"""
Music agent.

Generates BGM direction, a Suno-ready music prompt, and editing notes
based on the selected script, final topic, requested duration, tone,
and output language.
"""

from app.schemas.state import ContentStudioState


def music_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate music direction and editing notes for the short-form video.

    This agent does not create music directly. Instead, it generates a
    production-ready music prompt for tools such as Suno, along with
    editing notes aligned with the requested short-form duration.

    Priority of topic source:
    1. final_topic_suggestion
    2. normalized_topic
    3. raw topic fallback

    Priority of script source:
    1. revised_script
    2. best_script

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the music_output field.
    """
    request = state["request"]
    language = request.get("language", "en")
    raw_topic = request["topic"]
    tone = request["tone"]
    duration_sec = request.get("duration_sec", 30)

    director_brief = state.get("director_brief") or {}
    normalized_topic = director_brief.get("normalized_topic", raw_topic)
    final_topic = state.get("final_topic_suggestion") or normalized_topic or raw_topic

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    script_text = script_source.get("script", "")

    if duration_sec == 15:
        script_mood = script_text[:140] if script_text else "short, punchy, high-retention social content"
    elif duration_sec == 30:
        script_mood = script_text[:220] if script_text else "short-form, trendy, visually engaging social content"
    elif duration_sec == 45:
        script_mood = script_text[:320] if script_text else "slightly more detailed short-form social content with momentum"
    else:  # 60 sec
        script_mood = script_text[:420] if script_text else "structured social video content with clear pacing and stronger narrative flow"

    if language == "ko":
        if duration_sec == 15:
            bgm_direction = "짧고 강한 훅 중심의 경쾌한 숏폼 배경음악"
            editing_notes = [
                "첫 1~2초 안에 강한 비트 포인트 배치",
                "매우 짧고 빠른 컷 중심으로 편집",
                "핵심 자막만 크게 보여주기",
                "후반부는 빠르게 CTA로 연결",
            ]
        elif duration_sec == 30:
            bgm_direction = "경쾌하고 세련되며 숏폼 편집에 잘 맞는 현대적인 배경음악"
            editing_notes = [
                "첫 3초에 강한 비트 포인트 배치",
                "장면 전환마다 리듬감 있는 컷 편집",
                "큰 중앙 자막과 빠른 점프컷 사용",
                "2~3초마다 시각적 변화 주기",
                "엔드카드 직전에 볼륨을 살짝 낮춰 CTA를 강조",
            ]
        elif duration_sec == 45:
            bgm_direction = "트렌디한 리듬감과 적당한 전개감을 갖춘 숏폼 배경음악"
            editing_notes = [
                "도입부에 강한 훅 비트 배치",
                "중반부에 약한 변주를 넣어 지루함 방지",
                "핵심 포인트마다 자막과 컷 전환 강조",
                "후반부는 정리와 CTA가 자연스럽게 이어지도록 볼륨 조절",
            ]
        else:  # 60 sec
            bgm_direction = "몰입감을 유지하면서도 설명형 숏폼에 어울리는 리듬 중심 배경음악"
            editing_notes = [
                "초반 훅 이후에도 리듬이 너무 단조롭지 않게 구성",
                "중간 전개 구간에서 너무 과한 드롭은 피하기",
                "정보 전달 구간은 내레이션이 잘 들리도록 BGM 밀도 조절",
                "마지막 CTA 직전에는 살짝 정리되는 흐름 만들기",
            ]

        suno_prompt = (
            f"{final_topic}을 주제로 한 {duration_sec}초 숏폼 영상용 배경음악을 만들어줘. "
            f"톤은 '{tone}'이고, 전체 분위기는 트렌디하고 세련되며 에너지가 있어야 해. "
            f"과하게 웅장하지 말고, 짧은 영상 편집에 잘 맞도록 리듬감 있고 임팩트 있게 구성해줘. "
            f"보컬 없이, 자막과 내레이션 편집에 잘 어울리게 만들어줘. "
            f"오프닝 훅의 분위기는 '{hook}'이고, 영상 스크립트의 무드는 다음과 같아: {script_mood}"
        )

        output = {
            "agent_name": "music",
            "summary": "BGM 방향성과 Suno 프롬프트 생성",
            "bgm_direction": bgm_direction,
            "suno_prompt": suno_prompt,
            "editing_notes": editing_notes,
            "hook_reference": hook,
            "final_topic_reference": final_topic,
            "duration_reference": duration_sec,
        }

    else:
        if duration_sec == 15:
            bgm_direction = "short, punchy, high-energy background music optimized for quick retention"
            editing_notes = [
                "Hit the first beat immediately",
                "Use very fast cuts with minimal dead space",
                "Keep captions bold and minimal",
                "Move quickly into the CTA",
            ]
        elif duration_sec == 30:
            bgm_direction = "upbeat, modern, stylish, and optimized for short-form retention"
            editing_notes = [
                "Hit a strong beat within the first 3 seconds",
                "Use quick cuts synced to rhythm changes",
                "Add large center captions for key hook moments",
                "Keep visual changes every 2-3 seconds",
                "Lower the BGM slightly before the CTA for clarity",
            ]
        elif duration_sec == 45:
            bgm_direction = "modern and rhythmic background music with enough variation for mid-length short-form pacing"
            editing_notes = [
                "Open with a strong hook beat",
                "Introduce a subtle variation in the mid section",
                "Use pacing shifts to support key points",
                "Reduce intensity slightly before the CTA",
            ]
        else:  # 60 sec
            bgm_direction = "rhythmic, clean, and sustained background music for explanatory short-form storytelling"
            editing_notes = [
                "Avoid making the track feel repetitive across a full minute",
                "Keep the BGM supportive rather than overpowering during explanation-heavy sections",
                "Use small energy changes across segments",
                "Let the final CTA land clearly with a lighter ending section",
            ]

        suno_prompt = (
            f"Create a {duration_sec}-second background track for a short-form video about {final_topic}. "
            f"Tone: {tone}. "
            f"The track should feel modern, energetic, stylish, and highly editable for social video. "
            f"Avoid overly cinematic arrangements. Keep it clean, punchy, rhythmic, and suitable for fast cuts. "
            f"No vocals. Make it easy to edit around captions and voiceover. "
            f"The opening hook mood is '{hook}'. "
            f"The script mood is: {script_mood}"
        )

        output = {
            "agent_name": "music",
            "summary": "Generated BGM direction, Suno prompt, and editing notes",
            "bgm_direction": bgm_direction,
            "suno_prompt": suno_prompt,
            "editing_notes": editing_notes,
            "hook_reference": hook,
            "final_topic_reference": final_topic,
            "duration_reference": duration_sec,
        }

    return {
        "music_output": output,
    }