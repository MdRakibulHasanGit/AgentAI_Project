"""
Requirement Analysis Agent.

Turns free-text (Bangla/English/Banglish) user input into a structured
Requirement. Tries the LLM first (better at nuanced budget/use-case
extraction); falls back to the rule-based parser if no API key is set
or the call fails, so the pipeline never breaks.
"""
import json
from utils import llm_client
from utils.models import Requirement, parse_requirement_rule_based
from utils.logger import TRACE

AGENT = "requirement_agent"

SYSTEM_PROMPT = """You are a Requirement Analysis Agent for a laptop shopping assistant.
Extract a structured requirement from the user's free text (Bangla/English/Banglish).
Respond ONLY with JSON, no prose, matching this schema:
{"category": "laptop", "budget": <int or null, in BDT taka>, "use_case": [strings from: gaming, programming, video_editing, design, office, general],
 "priorities": {"CPU": "low|medium|high", "RAM": "low|medium|high", "GPU": "low|medium|high", "battery": "low|medium|high"},
 "min_ram_gb": <int>, "min_storage_gb": <int>}"""


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    TRACE.log("agent_start", AGENT, "Parsing user requirement from free text")

    text = state["user_text"]
    previous = Requirement.from_dict(state["requirement"]) if state.get("requirement") else None
    requirement = None

    if llm_client.is_available():
        try:
            raw = llm_client.ask_text(AGENT, SYSTEM_PROMPT, text, max_tokens=400)
            raw = raw.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)
            requirement = Requirement.from_dict(parsed)
            TRACE.log("message", AGENT, "Parsed requirement via LLM", payload=parsed)
        except Exception as e:
            TRACE.log("message", AGENT, f"LLM parse failed, falling back to rules: {e}", level="warn")
            requirement = None

    if requirement is None:
        requirement = parse_requirement_rule_based(text, previous)
        TRACE.log("message", AGENT, "Parsed requirement via rule-based fallback", payload=requirement.to_dict())

    TRACE.set_status(AGENT, "done")
    TRACE.log("agent_end", AGENT, f"Requirement: {requirement.to_dict()}")

    return {
        "requirement": requirement.to_dict(),
        "messages": [{"from": AGENT, "to": "supervisor", "content": f"Understood requirement: {requirement.to_dict()}"}],
        "next_agent": "web_search_agent",
    }
