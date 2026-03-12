"""
LLM client abstraction layer for Anthropic Claude models.

This module provides simple helper functions for invoking Claude models
from different agents within the Content Studio multi-agent system.
"""

import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        f"ANTHROPIC_API_KEY not found. Expected it in: {ENV_PATH}"
    )

client = Anthropic(api_key=ANTHROPIC_API_KEY)


def generate_with_sonnet(prompt: str) -> str:
    """
    Generate text using Claude Sonnet.

    Args:
        prompt: Prompt text.

    Returns:
        Generated response text.
    """

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.content[0].text


def generate_with_haiku(prompt: str) -> str:
    """
    Generate text using Claude Haiku.

    Args:
        prompt: Prompt text.

    Returns:
        Generated response text.
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.content[0].text