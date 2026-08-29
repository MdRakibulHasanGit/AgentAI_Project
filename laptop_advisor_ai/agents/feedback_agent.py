"""
Feedback / Re-planning Agent.

Handles follow-up turns (turn > 1): merges the new message into the
EXISTING requirement instead of discarding prior context, then routes
back into the pipeline from the RAG stage (the requirement is already
known, no need to re-run the from-scratch Requirement Analysis Agent).
This is what makes "battery aro valo lagbe" work as a refinement rather
than a restart.
"""
from utils import llm_client
from utils.models import Requirement, parse_requirement_rule_based
from utils.logger import TRACE

AGENT = "feedback_agent"

SYSTEM_PROMPT = """You are a Feedback / Re-planning Agent. The user already has a
structured requirement (given below) and just sent a follow-up message. Merge the
new information into the existing requirement -- do NOT discard existing fields
unless the user explicitly contradicts them. Respond ONLY with the full updated
JSON requirement, same schema as before."""


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    text = state["user_text"]
    previous = Requirement.from_dict(state.get("requirement", {}))

    updated = None
    if llm_client.is_available():
        try:
            import json
            user_prompt = f"Existing requirement: {previous.to_dict()}\nFollow-up message: {text}"
            raw = llm_client.ask_text(AGENT, SYSTEM_PROMPT, user_prompt, max_tokens=400).strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            updated = Requirement.from_dict(json.loads(raw))
        except Exception as e:
            TRACE.log("message", AGENT, f"LLM merge failed, using rule-based merge: {e}", level="warn")

    if updated is None:
        updated = parse_requirement_rule_based(text, previous)

    TRACE.set_status(AGENT, "done")
    TRACE.log("agent_end", AGENT, f"Merged requirement: {updated.to_dict()}")

    return {
        "requirement": updated.to_dict(),
        "scored_candidates": [],
        "critic_verdict": {},
        "revision_count": 0,
        "recommendation": {},
        "messages": [{"from": AGENT, "to": "supervisor", "content": f"Merged follow-up into requirement: {updated.to_dict()}"}],
        "next_agent": "rag_agent",
    }
