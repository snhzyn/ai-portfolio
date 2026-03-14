"""
Storyboard agent.

Generates a production-ready storyboard package for short-form video content
by decomposing the selected script into scene-level visual beats.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_haiku
from app.services.korean_text_utils import object_particle, polish_topic_for_visual


def _get_scene_ranges(duration_sec: int) -> list[str]:
    if duration_sec == 15:
        return ["0-2s", "2-5s", "5-11s", "11-15s"]
    if duration_sec == 30:
        return ["0-3s", "3-8s", "8-18s", "18-26s", "26-30s"]
    if duration_sec == 45:
        return ["0-3s", "3-8s", "8-18s", "18-28s", "28-38s", "38-45s"]
    if duration_sec == 60:
        return ["0-4s", "4-10s", "10-20s", "20-32s", "32-44s", "44-54s", "54-60s"]
    return ["0-3s", "3-8s", "8-18s", "18-26s", "26-30s"]


def _fallback_storyboard(
    language: str,
    duration_sec: int,
    visual_topic: str,
    hook: str,
    script_text: str,
    cta: str,
) -> dict:
    ranges = _get_scene_ranges(duration_sec)

    if language == "ko":
        scenes = [
            {
                "scene": 1,
                "time_range": ranges[0],
                "visual": f"{object_particle(visual_topic)} 상징적으로 보여주는 강한 오프닝 컷",
                "voiceover": hook or f"{visual_topic}, 왜 주목받고 있을까요?",
                "on_screen_text": "강한 오프닝",
            },
            {
                "scene": 2,
                "time_range": ranges[1],
                "visual": "관련 자료와 분위기 컷",
                "voiceover": script_text[:60] if script_text else "핵심 내용을 빠르게 시작합니다.",
                "on_screen_text": "장면 전개",
            },
            {
                "scene": 3,
                "time_range": ranges[2],
                "visual": "핵심 설명 장면",
                "voiceover": script_text[60:140] if len(script_text) > 60 else "핵심 포인트를 전달합니다.",
                "on_screen_text": "핵심 포인트",
            },
            {
                "scene": 4,
                "time_range": ranges[3],
                "visual": "맥락 확장 장면",
                "voiceover": script_text[140:220] if len(script_text) > 140 else "이 장면이 의미를 확장합니다.",
                "on_screen_text": "왜 중요할까?",
            },
            {
                "scene": 5,
                "time_range": ranges[4],
                "visual": "엔드카드 또는 CTA 컷",
                "voiceover": cta or "여러분 생각도 댓글로 남겨주세요.",
                "on_screen_text": "당신의 생각은?",
            },
        ]

        return {
            "agent_name": "storyboard",
            "summary": "숏폼 영상용 스토리보드 및 샷 리스트 생성",
            "scenes": scenes,
            "editing_style": "빠른 점프컷, 큰 중앙 자막, 첫 3초 강한 훅 중심 구성",
        }

    scenes = [
        {
            "scene": 1,
            "time_range": ranges[0],
            "visual": f"Strong opening visual related to {visual_topic}",
            "voiceover": hook or f"Why is {visual_topic} getting attention?",
            "on_screen_text": "STRONG HOOK",
        },
        {
            "scene": 2,
            "time_range": ranges[1],
            "visual": "Context setup visuals",
            "voiceover": script_text[:60] if script_text else "Here’s the setup.",
            "on_screen_text": "SETUP",
        },
        {
            "scene": 3,
            "time_range": ranges[2],
            "visual": "Main explanatory beat",
            "voiceover": script_text[60:140] if len(script_text) > 60 else "Here’s the key point.",
            "on_screen_text": "KEY POINT",
        },
        {
            "scene": 4,
            "time_range": ranges[3],
            "visual": "Context expansion",
            "voiceover": script_text[140:220] if len(script_text) > 140 else "Why it matters.",
            "on_screen_text": "WHY IT MATTERS",
        },
        {
            "scene": 5,
            "time_range": ranges[4],
            "visual": "End card or CTA frame",
            "voiceover": cta or "What do you think? Let me know in the comments.",
            "on_screen_text": "YOUR TAKE?",
        },
    ]

    return {
        "agent_name": "storyboard",
        "summary": "Generated storyboard and shot list for short-form video",
        "scenes": scenes,
        "editing_style": "fast cuts, large captions, high-retention pacing, strong first-3-second hook",
    }


def _normalize_scenes(
    scenes: list,
    scene_ranges: list[str],
    language: str,
) -> list[dict]:
    """
    Normalize imperfect LLM scene output into a stable structure.
    """
    normalized = []
    expected_count = len(scene_ranges)

    for idx in range(expected_count):
        raw_scene = scenes[idx] if idx < len(scenes) and isinstance(scenes[idx], dict) else {}

        normalized.append(
            {
                "scene": idx + 1,
                "time_range": raw_scene.get("time_range", scene_ranges[idx]),
                "visual": raw_scene.get(
                    "visual",
                    "관련 장면을 보여주는 컷" if language == "ko" else "Supporting visual cut",
                ),
                "voiceover": raw_scene.get(
                    "voiceover",
                    "핵심 내용을 전달합니다." if language == "ko" else "Deliver the key point.",
                ),
                "on_screen_text": raw_scene.get(
                    "on_screen_text",
                    "핵심 장면" if language == "ko" else "KEY MOMENT",
                ),
            }
        )

    return normalized


def storyboard_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate a storyboard and shot list for the video package.
    """
    request = state["request"]
    language = request.get("language", "en")
    raw_topic = request["topic"]
    duration_sec = request.get("duration_sec", 30)

    director_brief = state.get("director_brief") or {}
    research_output = state.get("research_output") or {}

    normalized_topic = director_brief.get("normalized_topic", raw_topic)
    core_angle = director_brief.get("core_angle", raw_topic)
    audience_frame = director_brief.get("audience_frame", request.get("audience", "general audience"))
    content_goal = director_brief.get("content_goal", "")
    final_topic = state.get("final_topic_suggestion") or normalized_topic or raw_topic
    visual_topic = polish_topic_for_visual(final_topic)

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    script_text = script_source.get("script", "")
    cta = script_source.get("cta", "")

    key_claims = research_output.get("key_claims", [])
    supporting_examples = research_output.get("supporting_examples", [])
    recommended_focus = research_output.get("recommended_focus", core_angle)

    scene_ranges = _get_scene_ranges(duration_sec)
    scene_count = len(scene_ranges)

    if language == "ko":
        prompt = f"""
You are the Storyboard Agent for an AI short-form content studio.

Your task is to convert the selected script into a production-ready storyboard.

Important:
- Use the selected script as the main source of truth.
- Follow the actual hook, script flow, and CTA.
- Avoid generic labels like "핵심 포인트" unless absolutely necessary.
- Each scene should reflect a real moment from the script.
- Match the requested scene count exactly: {scene_count}
- Use these exact time ranges: {scene_ranges}
- Keep on-screen text short, visual, and edit-friendly.

Request:
- Final topic: {final_topic}
- Core angle: {core_angle}
- Audience frame: {audience_frame}
- Content goal: {content_goal}
- Duration: {duration_sec}
- Hook: {hook}
- Script: {script_text}
- CTA: {cta}

Research support:
- Key claims: {key_claims}
- Supporting examples: {supporting_examples}
- Recommended focus: {recommended_focus}

Return valid JSON only.

Output schema:
{{
  "scenes": [
    {{
      "scene": 1,
      "time_range": "0-3s",
      "visual": "구체적인 장면 설명",
      "voiceover": "해당 장면용 짧은 내레이션",
      "on_screen_text": "짧은 자막"
    }}
  ],
  "editing_style": "한 줄 편집 스타일"
}}
"""
    else:
        prompt = f"""
You are the Storyboard Agent for an AI short-form content studio.

Your task is to convert the selected script into a production-ready storyboard.

Important:
- Use the selected script as the main source of truth.
- Follow the actual hook, script flow, and CTA.
- Avoid generic labels unless absolutely necessary.
- Each scene should reflect a real moment from the script.
- Match the requested scene count exactly: {scene_count}
- Use these exact time ranges: {scene_ranges}
- Keep on-screen text short, visual, and edit-friendly.

Request:
- Final topic: {final_topic}
- Core angle: {core_angle}
- Audience frame: {audience_frame}
- Content goal: {content_goal}
- Duration: {duration_sec}
- Hook: {hook}
- Script: {script_text}
- CTA: {cta}

Research support:
- Key claims: {key_claims}
- Supporting examples: {supporting_examples}
- Recommended focus: {recommended_focus}

Return valid JSON only.

Output schema:
{{
  "scenes": [
    {{
      "scene": 1,
      "time_range": "0-3s",
      "visual": "Specific visual description",
      "voiceover": "Short line for that scene",
      "on_screen_text": "Short caption"
    }}
  ],
  "editing_style": "One-line editing style"
}}
"""

    try:
        response = generate_with_haiku(prompt)
        parsed = parse_json_response(response)

        scenes = parsed.get("scenes", [])
        if not isinstance(scenes, list):
            scenes = []

        normalized_scenes = _normalize_scenes(scenes, scene_ranges, language)

        editing_style = parsed.get(
            "editing_style",
            "빠른 점프컷, 큰 중앙 자막, 첫 3초 강한 훅 중심 구성"
            if language == "ko"
            else "fast cuts, large captions, high-retention pacing, strong first-3-second hook",
        )

        output = {
            "agent_name": "storyboard",
            "summary": "숏폼 영상용 스토리보드 및 샷 리스트 생성"
            if language == "ko"
            else "Generated storyboard and shot list for short-form video",
            "scenes": normalized_scenes,
            "editing_style": editing_style,
        }

    except Exception:
        output = _fallback_storyboard(
            language=language,
            duration_sec=duration_sec,
            visual_topic=visual_topic,
            hook=hook,
            script_text=script_text,
            cta=cta,
        )

    return {
        "storyboard_output": output,
    }