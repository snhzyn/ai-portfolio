"""
Utilities for handling output language instructions.
"""


def get_language_instruction(language: str) -> str:
    """
    Return a prompt instruction for the desired output language.

    Args:
        language: Language code such as 'en' or 'ko'.

    Returns:
        Prompt instruction string.
    """
    if language == "ko":
        return (
            "Write the output in Korean. "
            "Use natural Korean suitable for short-form video narration and captions."
        )

    return (
        "Write the output in English. "
        "Use natural English suitable for short-form video narration and captions."
    )