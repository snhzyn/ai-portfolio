"""
Script generation agent.

Creates multiple candidate scripts for short-form video content.
"""

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
import json


def script_node(state: ContentStudioState) -> ContentStudioState:
    """
    Generate multiple short-form script variants.
    """

    request = state["request"]
    topic = request["topic"]

    prompt = f"""
You are a short-form video scriptwriter.

Topic: {topic}
Platform: {request["platform"]}
Audience: {request["audience"]}
Tone: {request["tone"]}
Duration: {request["duration_sec"]} seconds

Generate 3 different short-form video scripts.

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

Return JSON:

{{
 "scripts":[
   {{
     "hook":"...",
     "script":"...",
     "cta":"..."
   }},
   ...
 ]
}}
"""

    response = generate_with_sonnet(prompt)

    try:
        parsed = json.loads(response)
        scripts = parsed["scripts"]
    except Exception:
        scripts = [
            {
                "hook": f"Why is {topic} trending?",
                "script": f"People keep talking about {topic}. Here's why.",
                "cta": "What do you think?"
            }
        ]

    return {
        "script_candidates": scripts,
        "logs": state.get("logs", []) + [{"node": "script", "status": "completed"}],
    }