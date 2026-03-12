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

    Args:
        topic: Raw topic string from the API request.
        language: Output language code, such as "en" or "ko".

    Returns:
        A normalized topic string suitable for downstream content generation.
    """
    normalized = topic.strip()

    if language == "en":
        lowered = normalized.lower()

        for prefix in ["why ", "how ", "what ", "when "]:
            if lowered.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                lowered = normalized.lower()
                break

        normalized = normalized.rstrip(" ?")

        trend_match = re.match(r"(.+?)\s+is trending in\s+(\d{4})$", normalized, flags=re.IGNORECASE)
        if trend_match:
            subject = trend_match.group(1).strip()
            year = trend_match.group(2).strip()
            return f"{subject} in {year}"

        popular_match = re.match(r"(.+?)\s+is popular in\s+(\d{4})$", normalized, flags=re.IGNORECASE)
        if popular_match:
            subject = popular_match.group(1).strip()
            year = popular_match.group(2).strip()
            return f"{subject} in {year}"

        normalized = " ".join(normalized.split())
        return normalized

    if language == "ko":
        text = normalized.replace("?", "").strip()
        text = re.sub(r"\s+", " ", text)

        # 패턴 1: "X은/는 왜 Y할까/일까/일어났을까요/유행하는가" -> "X의 Y 배경/이유"
        patterns = [
            (r"^(?P<subject>.+?)(?:은|는)\s*왜\s*(?P<predicate>일어났을까요|일어났을까|일어났는가)$", r"\g<subject>의 발발 배경"),
            (r"^(?P<subject>.+?)(?:은|는)\s*왜\s*(?P<predicate>생겼을까요|생겼을까|생겼는가)$", r"\g<subject>의 형성 배경"),
            (r"^(?P<subject>.+?)(?:은|는)\s*왜\s*(?P<predicate>유행하는가|유행할까|유행할까요)$", r"\g<subject> 유행 배경"),
            (r"^(?P<subject>.+?)(?:이|가)\s*왜\s*(?P<predicate>유행하는가|유행할까|유행할까요)$", r"\g<subject> 유행 배경"),
            (r"^(?P<subject>.+?)(?:은|는)\s*왜\s*(?P<predicate>중요할까|중요할까요|중요한가)$", r"\g<subject>의 중요성"),
            (r"^(?P<subject>.+?)(?:은|는)\s*왜\s*(?P<predicate>문제일까|문제일까요|문제인가)$", r"\g<subject>의 쟁점"),
        ]

        for pattern, replacement in patterns:
            if re.match(pattern, text):
                return re.sub(pattern, replacement, text).strip()

        # 패턴 2: 문장 앞의 "왜 " 제거
        if text.startswith("왜 "):
            text = text[2:].strip()

        # 패턴 3: 끝 질문형 어미 제거
        endings_to_strip = [
            "일어났을까요",
            "일어났을까",
            "일어났는가",
            "유행하는가",
            "유행할까",
            "유행할까요",
            "선호할까",
            "선호할까요",
            "선호하는가",
            "중요할까",
            "중요할까요",
            "중요한가",
            "문제일까",
            "문제일까요",
            "문제인가",
            "무엇일까",
            "무엇일까요",
            "뭘까",
            "왜일까",
        ]

        for ending in endings_to_strip:
            if text.endswith(ending):
                text = text[: -len(ending)].strip()
                break

        # 패턴 4: 한국어 질문형을 설명형으로 마무리
        if text.endswith("전쟁"):
            return f"{text}의 배경"
        if text.endswith("갈등"):
            return f"{text}의 배경"
        if text.endswith("트렌드"):
            return text
        if text.endswith("러닝"):
            return f"{text} 트렌드"
        if text.endswith("중고거래"):
            return f"{text} 선호 현상"
        if text.endswith("명상"):
            return f"{text} 확산 배경"

        # fallback
        text = text.replace("  ", " ").strip()
        return text

    return normalized


def director_node(state: ContentStudioState) -> ContentStudioState:
    """
    Build the high-level creative brief and planned agents.

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