"""
QA agent.

Evaluates available writer outputs, selects the best script candidate,
and produces a QA package for downstream revision.
"""

from app.schemas.state import ContentStudioState
from app.services.json_utils import parse_json_response
from app.services.llm_client import generate_with_haiku


def _collect_candidates(state: ContentStudioState) -> list[dict]:
    """
    Collect script candidates from writer outputs only.
    This avoids duplicate candidates caused by concurrent state merging.
    """
    candidates = []

    writer_fast = state.get("writer_fast_output")
    writer_story = state.get("writer_story_output")
    writer_viral = state.get("writer_viral_output")

    for candidate in [writer_fast, writer_story, writer_viral]:
        if isinstance(candidate, dict) and candidate.get("hook") and candidate.get("script"):
            candidates.append(candidate)

    return candidates


def qa_node(state: ContentStudioState) -> ContentStudioState:
    """
    Evaluate script candidates and choose the strongest one.
    """
    request = state["request"]
    language = request.get("language", "en")
    audience = request.get("audience", "general audience")
    platform = request.get("platform", "youtube_shorts")
    duration_sec = request.get("duration_sec", 30)

    candidates = _collect_candidates(state)

    if not candidates:
        fallback_best = {
            "style": "fast",
            "hook": request.get("topic", ""),
            "script": request.get("topic", ""),
            "cta": "댓글로 의견 남겨주세요." if language == "ko" else "Let me know in the comments.",
        }
        return {
            "script_candidates": [],
            "best_script": fallback_best,
            "qa_output": {
                "selected_script": 0,
                "reason": "Fallback selection due to missing writer outputs.",
                "quality_score": 5.0,
            },
        }

    candidate_text = []
    for idx, candidate in enumerate(candidates):
        candidate_text.append(
            f"""
Candidate {idx}
Style: {candidate.get("style", "")}
Hook: {candidate.get("hook", "")}
Script: {candidate.get("script", "")}
CTA: {candidate.get("cta", "")}
""".strip()
        )

    joined_candidates = "\n\n".join(candidate_text)

    prompt = f"""
You are the QA Agent for an AI short-form content studio.

Your task is to evaluate the script candidates and choose the strongest one.

Context:
- Audience: {audience}
- Platform: {platform}
- Duration: {duration_sec} seconds
- Language: {language}

Evaluation criteria:
1. Hook strength
2. Clarity
3. Pacing
4. Platform suitability
5. Audience fit

Candidates:
{joined_candidates}

Return valid JSON only.
Do not include markdown fences.

Output schema:
{{
  "selected_script": 0,
  "reason": "Why this script is best",
  "quality_score": 8.7
}}
"""

    try:
        response = generate_with_haiku(prompt)
        parsed = parse_json_response(response)

        selected_idx = int(parsed.get("selected_script", 0))
        if selected_idx < 0 or selected_idx >= len(candidates):
            selected_idx = 0

        qa_output = {
            "selected_script": selected_idx,
            "reason": parsed.get("reason", "Selected based on overall short-form strength."),
            "quality_score": float(parsed.get("quality_score", 7.5)),
        }
    except Exception:
        selected_idx = 0
        qa_output = {
            "selected_script": 0,
            "reason": "Fallback selection due to parsing failure.",
            "quality_score": 7.5,
        }

    return {
        "script_candidates": candidates,
        "best_script": candidates[selected_idx],
        "qa_output": qa_output,
    }