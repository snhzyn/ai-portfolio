"""
Utilities for extracting JSON objects from model responses.
"""

import json
import re
from typing import Any


def parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from a model response.

    This helper handles cases where the model wraps JSON in markdown
    fences or adds extra text around the JSON payload.

    Args:
        text: Raw model response text.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    cleaned = text.strip()

    # Remove markdown fences if present.
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Try direct parse first.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object block.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from model response.")