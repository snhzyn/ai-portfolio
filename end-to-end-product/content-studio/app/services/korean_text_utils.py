"""
Utilities for basic Korean particle handling and phrase polishing.
"""


def _has_batchim(text: str) -> bool:
    """
    Check whether the last Korean syllable has a final consonant (batchim).

    Args:
        text: Input text.

    Returns:
        True if the last Korean syllable has batchim, otherwise False.
    """
    if not text:
        return False

    last_char = text.strip()[-1]
    code = ord(last_char)

    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0

    return False


def topic_particle(text: str) -> str:
    """
    Attach the correct Korean topic particle 은/는.

    Args:
        text: Input noun phrase.

    Returns:
        Text with 은 or 는 attached.
    """
    return f"{text}{'은' if _has_batchim(text) else '는'}"


def subject_particle(text: str) -> str:
    """
    Attach the correct Korean subject particle 이/가.

    Args:
        text: Input noun phrase.

    Returns:
        Text with 이 or 가 attached.
    """
    return f"{text}{'이' if _has_batchim(text) else '가'}"


def object_particle(text: str) -> str:
    """
    Attach the correct Korean object particle 을/를.

    Args:
        text: Input noun phrase.

    Returns:
        Text with 을 or 를 attached.
    """
    return f"{text}{'을' if _has_batchim(text) else '를'}"


def with_particle(text: str, kind: str) -> str:
    """
    Attach the requested Korean particle.

    Supported kinds:
    - topic: 은/는
    - subject: 이/가
    - object: 을/를

    Args:
        text: Input noun phrase.
        kind: Particle type.

    Returns:
        Text with the requested particle attached.
    """
    if kind == "topic":
        return topic_particle(text)
    if kind == "subject":
        return subject_particle(text)
    if kind == "object":
        return object_particle(text)
    return text


def polish_topic_for_visual(text: str) -> str:
    """
    Lightly polish a topic phrase for visual description usage.

    Args:
        text: Topic phrase.

    Returns:
        A slightly cleaner visual-friendly phrase.
    """
    cleaned = " ".join(text.strip().split())
    return cleaned