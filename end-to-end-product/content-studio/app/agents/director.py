"""
Director agent.

Responsible for interpreting the user request and deciding
which specialist agents should be activated.
"""

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
import json


def director_node(state: ContentStudioState) -> ContentStudioState:
    """
    Analyze the request and decide which agents should run.
    """

    request = state["request"]

    prompt = f"""
You are the Director of an AI Content Production Studio.

Analyze the request and decide which specialist agents should run.

Topic: {request["topic"]}
Platform: {request["platform"]}
Audience: {request["audience"]}
Tone: {request["tone"]}

Available agents:
- research
- script
- storyboard
- title_thumbnail
- music

Return JSON:

{{
 "core_angle": "...",
 "agent_plan": ["script","storyboard","title_thumbnail","music"]
}}
"""

    response = generate_with_sonnet(prompt)

    try:
        parsed = json.loads(response)
    except Exception:
        parsed = {
            "core_angle": request["topic"],
            "agent_plan": ["script", ] # "storyboard", "title_thumbnail", "music"
        }

    return {
        "director_brief": {
            "core_angle": parsed["core_angle"]
        },
        "planned_agents": parsed["agent_plan"],
        "logs": state.get("logs", []) + [{"node": "director", "status": "completed"}],
    }