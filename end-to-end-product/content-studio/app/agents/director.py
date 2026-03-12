"""
Director agent.

Responsible for interpreting the user request and deciding
which specialist agents should be activated.
"""

from app.schemas.state import ContentStudioState
from app.services.llm_client import generate_with_sonnet
from app.services.json_utils import parse_json_response
import json


def director_node(state: ContentStudioState) -> ContentStudioState:
    """
    Analyze the content request and determine which specialist agents
    should run in the workflow.

    Args:
        state: The current LangGraph state.

    Returns:
        Updated state including the director brief and planned agents.
    """
    request = state["request"]

    prompt = f"""
You are the Director of an AI Content Production Studio.

Analyze the request and decide which specialist agents should run.

Topic: {request["topic"]}
Platform: {request["platform"]}
Audience: {request["audience"]}
Tone: {request["tone"]}
Duration: {request["duration_sec"]} seconds

Available agents:
- writer_fast
- writer_story
- writer_viral
- storyboard
- title_thumbnail
- music

Return valid JSON only.
Do not include markdown fences.
Do not include any explanation before or after the JSON.

{{
  "core_angle": "...",
  "agent_plan": ["writer_fast", "writer_story", "writer_viral"]
}}
"""

    # response = generate_with_sonnet(prompt)

    # try:
    #     parsed = parse_json_response(response)
    # except Exception:
    #     parsed = {
    #         "core_angle": request["topic"],
    #         "agent_plan": ["writer_fast", "writer_story", "writer_viral"],
    #     }

    return {
        "director_brief": {
            "core_angle": request["topic"],
        },
        "planned_agents": ["writer_fast", "writer_story", "writer_viral"],
    }

    # return {
    #     "director_brief": {
    #         "core_angle": parsed["core_angle"],
    #     },
    #     "planned_agents": parsed["agent_plan"],
    # }