"""
Director agent.

Responsible for interpreting the user request and deciding
which specialist agents should be activated.
"""

import re

from app.schemas.state import ContentStudioState


def _normalize_topic(topic: str, language: str) -> str:
    """
    Normalize a raw user topic into a cleaner production-friendly topic string.

    This helps prevent awkward downstream outputs such as:
    - "Why Why matcha is trending in 2026..."
    - "Why is everyone talking about matcha is trending in 2026?"

    Args:
        topic: Raw topic string from the API request.
        language: Output language code, such as "en" or "ko".

    Returns:
        A normalized topic string suitable for downstream content generation.
    """
    normalized = topic.strip()

    if language == "en":
        lowered = normalized.lower()

        # Remove leading question words.
        for prefix in ["why ", "how ", "what ", "when "]:
            if lowered.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                lowered = normalized.lower()
                break

        # Remove trailing question mark.
        normalized = normalized.rstrip(" ?")

        # Make common "X is trending in YEAR" patterns more title-friendly.
        trend_match = re.match(r"(.+?)\s+is trending in\s+(\d{4})$", normalized, flags=re.IGNORECASE)
        if trend_match:
            subject = trend_match.group(1).strip()
            year = trend_match.group(2).strip()
            return f"{subject} in {year}"

        # Make common "X is popular in YEAR" patterns more title-friendly.
        popular_match = re.match(r"(.+?)\s+is popular in\s+(\d{4})$", normalized, flags=re.IGNORECASE)
        if popular_match:
            subject = popular_match.group(1).strip()
            year = popular_match.group(2).strip()
            return f"{subject} in {year}"

        # Fallback cleanup.
        normalized = " ".join(normalized.split())
        return normalized

    if language == "ko":
        normalized = normalized.replace("?", "").strip()

        # Remove common Korean question framing.
        replacements = [
            ("왜 ", ""),
            ("이유", ""),
            ("무엇일까", ""),
            ("뭘까", ""),
            ("왜일까", ""),
        ]

        for old, new in replacements:
            normalized = normalized.replace(old, new)

        normalized = " ".join(normalized.split())
        return normalized

    return normalized


def director_node(state: ContentStudioState) -> ContentStudioState:
    """
    Build the high-level creative brief and planned agents.

    For now, this version uses a deterministic agent plan to keep
    the workflow stable during iteration. The director also creates
    a normalized topic string so downstream agents can generate more
    natural titles, hooks, storyboard lines, and prompts.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state containing the director brief and planned agents.
    """
    request = state["request"]
    language = request.get("language", "en")
    raw_topic = request["topic"]

    normalized_topic = _normalize_topic(raw_topic, language)

    return {
        "director_brief": {
            "core_angle": raw_topic,
            "normalized_topic": normalized_topic,
            "language": language,
        },
        "planned_agents": [
            "writer_fast",
            "writer_story",
            "writer_viral",
            "storyboard",
            "title_thumbnail",
            "music",
        ],
    }