"""
Supervisor Agent.

Not just a router-by-string -- this node makes the entry-point decision
(first turn -> full pipeline from Requirement Analysis; follow-up turn
-> Feedback/Re-planning Agent, preserving prior context) and is the
single place that decides overall workflow shape. Every specialized
agent still proposes its own `next_agent`, but the Supervisor is
consulted at the one branch point that needs global context (the
state's turn count), matching a true supervisor/worker topology rather
than a flat linear chain.
"""
from utils.logger import TRACE

AGENT = "supervisor"


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    turn = state.get("turn", 0) + 1
    is_followup = turn > 1 and bool(state.get("requirement"))

    entry = "feedback_agent" if is_followup else "requirement_agent"
    TRACE.log(
        "message", AGENT,
        f"Turn {turn}: routing to {entry} ({'follow-up, reusing prior requirement' if is_followup else 'fresh session'})",
    )
    TRACE.set_status(AGENT, "done")

    return {
        "turn": turn,
        "next_agent": entry,
        "messages": [{"from": AGENT, "to": entry, "content": f"Turn {turn} dispatched to {entry}"}],
    }


def route_after_supervisor(state: dict) -> str:
    return state["next_agent"]


def route_generic(state: dict) -> str:
    """Used for every agent node's outgoing conditional edge: each agent
    sets state['next_agent'] to the name of the node it wants control to
    pass to next (including back-edges like critic -> filter_compare, or
    forward edges to 'human_review' / 'end')."""
    return state.get("next_agent", "end")
