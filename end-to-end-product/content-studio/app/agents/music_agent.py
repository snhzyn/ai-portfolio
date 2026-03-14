"""
Music agent.

Designs background music direction and audio guidance for short-form video tools
such as InVideo or CapCut, based on the final script, hook, tone, duration,
and scene flow.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_haiku


def _build_music_cues_by_duration(language: str, duration_sec: int) -> list[str]:
    """
    Build fallback music cue structure by duration.
    """
    if language == "ko":
        if duration_sec == 15:
            return [
                "0-2초: 강한 신스 임팩트로 시작",
                "2-8초: 부드러운 앰비언트 패드와 가벼운 리듬 유지",
                "8-12초: 에너지 살짝 상승",
                "12-15초: CTA를 위해 볼륨을 자연스럽게 낮춤",
            ]
        if duration_sec == 30:
            return [
                "0-3초: 강한 신스 임팩트로 시작",
                "3-10초: 부드러운 앰비언트 패드와 가벼운 리듬",
                "10-20초: 리듬과 베이스를 살짝 강화",
                "20-27초: 감정 고조를 위한 신스 레이어 확장",
                "27-30초: CTA를 위해 볼륨을 자연스럽게 낮춤",
            ]
        if duration_sec == 45:
            return [
                "0-3초: 강한 신스 임팩트로 시작",
                "3-12초: 부드러운 앰비언트 패드와 가벼운 리듬",
                "12-24초: 리듬과 베이스를 강화하며 몰입감 형성",
                "24-36초: 감정 고조를 위한 신스 레이어 확장",
                "36-42초: 여운을 주는 안정 구간",
                "42-45초: CTA를 위해 볼륨을 낮춤",
            ]
        return [
            "0-4초: 강한 신스 임팩트로 시작",
            "4-15초: 부드러운 앰비언트 패드와 리듬 형성",
            "15-30초: 리듬과 베이스를 강화하며 본격 전개",
            "30-45초: 감정 고조를 위한 신스 레이어 확장",
            "45-54초: 여운과 공간감을 살리는 안정 구간",
            "54-60초: CTA 또는 엔딩을 위해 볼륨을 자연스럽게 낮춤",
        ]

    if duration_sec == 15:
        return [
            "0-2s: strong synth impact intro",
            "2-8s: soft ambient pads with light rhythm",
            "8-12s: slight energy lift",
            "12-15s: lower volume naturally for CTA",
        ]
    if duration_sec == 30:
        return [
            "0-3s: strong synth impact intro",
            "3-10s: soft ambient pads with light rhythm",
            "10-20s: slightly stronger bass and pulse",
            "20-27s: emotional lift with layered synths",
            "27-30s: volume reduction for CTA clarity",
        ]
    if duration_sec == 45:
        return [
            "0-3s: strong synth impact intro",
            "3-12s: soft ambient pads with light rhythm",
            "12-24s: stronger bass and pulse for forward motion",
            "24-36s: emotional lift with layered synths",
            "36-42s: brief settle section with space",
            "42-45s: lower volume for CTA clarity",
        ]
    return [
        "0-4s: strong synth impact intro",
        "4-15s: ambient pads and groove build",
        "15-30s: stronger bass and rhythmic pulse",
        "30-45s: emotional synth expansion",
        "45-54s: brief atmospheric settle section",
        "54-60s: lower volume naturally for ending or CTA",
    ]


def _fallback_music(language: str, final_topic: str, hook: str, duration_sec: int) -> dict:
    """
    Safe fallback when LLM generation fails.
    """
    music_cues = _build_music_cues_by_duration(language, duration_sec)

    if language == "ko":
        return {
            "agent_name": "music",
            "summary": "영상 생성 툴용 배경음악 방향성과 오디오 가이드 생성",
            "bgm_direction": "몽환적이고 미래적인 앰비언트 전자 음악",
            "music_cues": music_cues,
            "audio_style_notes": [
                "보컬 없는 배경음악",
                "미래적이고 세련된 질감",
                "숏폼 편집에 맞는 리듬감",
                "내레이션을 방해하지 않는 믹스",
            ],
            "editing_notes": [
                "첫 도입 구간에 오디오 임팩트 배치",
                "장면 전환에 맞춰 리듬 변화를 주기",
                "중반부에 에너지 살짝 상승",
                "마지막 CTA 또는 엔딩 직전 볼륨 다운",
            ],
            "hook_reference": hook,
            "final_topic_reference": final_topic,
            "duration_reference": duration_sec,
        }

    return {
        "agent_name": "music",
        "summary": "Generated background music direction and audio guidance for video tools",
        "bgm_direction": "dreamy futuristic ambient electronic background music",
        "music_cues": music_cues,
        "audio_style_notes": [
            "instrumental only",
            "futuristic and polished texture",
            "rhythmic enough for short-form editing",
            "mixed under narration cleanly",
        ],
        "editing_notes": [
            "place an audio impact in the opening section",
            "sync rhythm changes with scene cuts",
            "lift energy slightly in the middle",
            "lower volume before the CTA or ending",
        ],
        "hook_reference": hook,
        "final_topic_reference": final_topic,
        "duration_reference": duration_sec,
    }


def music_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate BGM direction and audio guidance for video generation tools.
    """
    request = state["request"]
    language = request.get("language", "en")
    tone = request.get("tone", "")
    duration_sec = request.get("duration_sec", 30)

    director_brief = state.get("director_brief") or {}
    final_topic = state.get("final_topic_suggestion") or director_brief.get("normalized_topic", "")

    revised_script = state.get("revised_script") or {}
    best_script = state.get("best_script") or {}
    script_source = revised_script or best_script

    hook = script_source.get("hook", "")
    script_text = script_source.get("script", "")

    storyboard = state.get("storyboard_output") or {}
    scenes = storyboard.get("scenes", [])

    scene_summary = "\n".join(
        [
            f"{s.get('time_range')} - {s.get('visual')}"
            for s in scenes
        ]
    )

    if language == "ko":
        prompt = f"""
You are the Music Agent for an AI short-form video production system.

Your job is generating audio direction that can be embedded directly into a video generation prompt
for tools like InVideo or CapCut.

You must output:
1. bgm_direction
2. music_cues
3. audio_style_notes
4. editing_notes

Context:

Final topic:
{final_topic}

Hook:
{hook}

Tone:
{tone}

Duration:
{duration_sec} seconds

Script:
{script_text}

Scene flow:
{scene_summary}

Requirements:
- This is a {duration_sec}-second vertical short-form video
- Music should support quick edits and strong short-form retention
- No vocals
- Avoid cinematic trailer style
- Prefer modern, social-media-friendly background music
- Match the emotional arc of the script
- Reflect the requested runtime in the music structure
- Output should help a video generation tool choose or create suitable BGM automatically

Return JSON only.

Schema:
{{
  "bgm_direction": "...",
  "music_cues": ["...", "...", "..."],
  "audio_style_notes": ["...", "...", "..."],
  "editing_notes": ["...", "...", "..."]
}}
"""
    else:
        prompt = f"""
You are the Music Agent for an AI short-form video production system.

Your job is generating audio direction that can be embedded directly into a video generation prompt
for tools like InVideo or CapCut.

Context:

Final topic:
{final_topic}

Hook:
{hook}

Tone:
{tone}

Duration:
{duration_sec} seconds

Script:
{script_text}

Scene flow:
{scene_summary}

Requirements:
- This is a {duration_sec}-second vertical short-form video
- Reflect the requested runtime in the music structure
- No vocals
- Match the emotional arc of the script
- Music should support fast social-video editing

Return JSON only.

Schema:
{{
  "bgm_direction": "...",
  "music_cues": ["...", "...", "..."],
  "audio_style_notes": ["...", "...", "..."],
  "editing_notes": ["...", "...", "..."]
}}
"""

    try:
        response = generate_with_haiku(prompt)
        parsed = parse_json_response(response)

        output = {
            "agent_name": "music",
            "summary": "영상 생성 툴용 배경음악 방향성과 오디오 가이드 생성"
            if language == "ko"
            else "Generated background music direction and audio guidance for video tools",
            "bgm_direction": parsed.get("bgm_direction", ""),
            "music_cues": parsed.get("music_cues", []),
            "audio_style_notes": parsed.get("audio_style_notes", []),
            "editing_notes": parsed.get("editing_notes", []),
            "hook_reference": hook,
            "final_topic_reference": final_topic,
            "duration_reference": duration_sec,
        }

    except Exception:
        output = _fallback_music(language, final_topic, hook, duration_sec)

    return {
        "music_output": output
    }