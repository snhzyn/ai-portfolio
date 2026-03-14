"""
Packaging agent.

Combines outputs from all agents into the final API response payload,
including an editor-ready brief and a video-tool-ready generation prompt.
"""

from app.schemas.state import ContentStudioState


def _build_video_generation_prompt(
    language: str,
    platform: str,
    audience: str,
    duration_sec: int,
    final_topic: str,
    tone: str,
    hook: str,
    revised_script: str,
    scenes: list[dict],
    thumbnail_text: list[str],
    cta: str,
    editing_notes: list[str],
    bgm_direction: str,
    music_cues: list[str],
    audio_style_notes: list[str],
) -> str:
    """
    Build a single prompt that can be pasted into a video generation tool
    or used as a master creative brief for video production.
    """
    scene_lines = []
    for scene in scenes:
        scene_lines.append(
            f"Scene {scene.get('scene')}: "
            f"{scene.get('time_range')} | "
            f"Visual: {scene.get('visual')} | "
            f"Voiceover: {scene.get('voiceover')} | "
            f"On-screen text: {scene.get('on_screen_text')}"
        )

    scenes_block = "\n".join(scene_lines)
    thumbnail_block = ", ".join(thumbnail_text) if thumbnail_text else ""
    editing_block = "\n".join(f"- {note}" for note in editing_notes) if editing_notes else ""
    music_cues_block = "\n".join(f"- {cue}" for cue in music_cues) if music_cues else ""
    audio_style_block = "\n".join(f"- {note}" for note in audio_style_notes) if audio_style_notes else ""

    if language == "ko":
        return f"""
{platform}용 {duration_sec}초 세로형 숏폼 영상을 제작해줘.

최종 주제:
{final_topic}

타깃 시청자:
{audience}

톤앤매너:
{tone}

오프닝 훅:
{hook}

최종 내레이션 스크립트:
{revised_script}

장면 구성:
{scenes_block}

추천 썸네일 문구:
{thumbnail_block}

배경음악 방향:
{bgm_direction}

오디오 스타일 가이드:
{audio_style_block}

장면별 음악/오디오 큐:
{music_cues_block}

마무리 CTA:
{cta}

편집 방향:
{editing_block}

제작 가이드:
- 세로형 9:16 비율
- 첫 3초에 강한 훅과 시각적 임팩트 강조
- 장면 전환은 빠르고 리듬감 있게
- 자막은 크고 짧고 직관적으로
- 전체적으로 트렌디하고 소셜미디어 친화적인 느낌
- 내레이션과 자막은 모두 한국어로 구성
- 배경음악은 보컬 없이 사용
- 배경음악은 내레이션을 방해하지 않도록 믹스
- CTA 직전에는 배경음악 볼륨을 자연스럽게 낮출 것
- 영상 생성 툴이 위 분위기에 맞는 음악을 자동 선택하거나 생성하도록 반영할 것
""".strip()

    return f"""
Create a {duration_sec}-second vertical short-form video for {platform}.

Final topic:
{final_topic}

Target audience:
{audience}

Tone and style:
{tone}

Opening hook:
{hook}

Final narration script:
{revised_script}

Scene plan:
{scenes_block}

Suggested thumbnail text:
{thumbnail_block}

Background music direction:
{bgm_direction}

Audio style guide:
{audio_style_block}

Scene-by-scene music and audio cues:
{music_cues_block}

Final CTA:
{cta}

Editing direction:
{editing_block}

Production guidelines:
- Use a vertical 9:16 format
- Make the first 3 seconds visually strong
- Keep cuts fast and rhythmically paced
- Use large, short, high-contrast captions
- Keep the overall style trendy, modern, and social-media-native
- Keep all narration and on-screen text in English
- Use instrumental background music only
- Mix the music under narration cleanly
- Lower music volume naturally before the CTA
- Let the video generation tool choose or create music that matches the above direction
""".strip()


def packaging_node(state: ContentStudioState) -> ContentStudioState:
    """
    Package all agent outputs into the final JSON structure.
    """
    request = state["request"]
    language = request.get("language", "en")
    platform = request.get("platform", "youtube_shorts")
    audience = request.get("audience", "general audience")
    tone = request.get("tone", "engaging, concise, platform-native")
    duration_sec = request.get("duration_sec", 30)

    director_brief = state.get("director_brief") or {}
    normalized_topic = director_brief.get("normalized_topic", request.get("topic", ""))
    final_topic = state.get("final_topic_suggestion") or normalized_topic or request.get("topic", "")

    revised_script_obj = state.get("revised_script") or {}
    best_script_obj = state.get("best_script") or {}
    final_script_obj = revised_script_obj or best_script_obj

    hook = final_script_obj.get("hook", "")
    revised_script = final_script_obj.get("script", "")
    cta = final_script_obj.get("cta", "")

    storyboard_output = state.get("storyboard_output") or {}
    scenes = storyboard_output.get("scenes", [])

    title_thumbnail_output = state.get("title_thumbnail_output") or {}
    thumbnail_text = title_thumbnail_output.get("thumbnail_text", [])

    music_output = state.get("music_output") or {}
    editing_notes = music_output.get("editing_notes", [])
    bgm_direction = music_output.get("bgm_direction", "")
    music_cues = music_output.get("music_cues", [])
    audio_style_notes = music_output.get("audio_style_notes", [])

    video_generation_prompt = _build_video_generation_prompt(
        language=language,
        platform=platform,
        audience=audience,
        duration_sec=duration_sec,
        final_topic=final_topic,
        tone=tone,
        hook=hook,
        revised_script=revised_script,
        scenes=scenes,
        thumbnail_text=thumbnail_text,
        cta=cta,
        editing_notes=editing_notes,
        bgm_direction=bgm_direction,
        music_cues=music_cues,
        audio_style_notes=audio_style_notes,
    )

    editor_brief = {
        "format": "vertical short-form video",
        "platform": platform,
        "duration_sec": duration_sec,
        "language": language,
        "topic": final_topic,
        "audience": audience,
        "tone": tone,
        "hook": hook,
        "narration_script": revised_script,
        "cta": cta,
        "scene_plan": scenes,
        "thumbnail_text": thumbnail_text,
        "bgm_direction": bgm_direction,
        "music_cues": music_cues,
        "audio_style_notes": audio_style_notes,
        "editing_notes": editing_notes,
    }

    final_json = {
        "creative_brief": director_brief,
        "research_package": state.get("research_output"),
        "final_topic_suggestion": state.get("final_topic_suggestion"),
        "script_candidates": state.get("script_candidates"),
        "best_script": state.get("best_script"),
        "revised_script": state.get("revised_script"),
        "writer_outputs": {
            "writer_fast": state.get("writer_fast_output"),
            "writer_story": state.get("writer_story_output"),
            "writer_viral": state.get("writer_viral_output"),
        },
        "storyboard_package": storyboard_output,
        "publish_package": title_thumbnail_output,
        "music_package": music_output,
        "qa_package": state.get("qa_output"),
        "editor_brief": editor_brief,
        "video_generation_prompt": video_generation_prompt,
    }

    return {
        "final_json": final_json,
    }