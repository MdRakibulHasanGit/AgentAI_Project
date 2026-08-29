"""
Human-in-the-loop node.

Uses LangGraph's native `interrupt()` to pause graph execution right
before the final report is written, and surfaces the pending
recommendation to the Streamlit UI. The UI's Approve/Reject/Retry
buttons resume the graph with `Command(resume=...)` on the same
thread_id, so the paused run continues exactly where it left off --
no state is lost.
"""
from langgraph.types import interrupt
from utils.logger import TRACE

AGENT = "human_review"


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    rec = state.get("recommendation", {})
    top = rec.get("top_pick")

    TRACE.log("hitl", AGENT, "Pausing for human approval before finalizing report")

    decision = interrupt({
        "kind": "approve_recommendation",
        "top_pick": top["product"]["name"] if top else None,
        "score": top["score"] if top else None,
        "message": rec.get("message", ""),
    })
    # `decision` is whatever value the caller passed via Command(resume=...)
    # e.g. {"action": "approve"} | {"action": "reject", "feedback": "..."} | {"action": "retry"}

    action = decision.get("action", "approve")
    TRACE.log("hitl", AGENT, f"Human decision received: {action}")
    TRACE.set_status(AGENT, "done")

    if action == "approve":
        return {"hitl_status": "approved", "next_agent": "report_agent",
                "messages": [{"from": "human", "to": AGENT, "content": "Approved"}]}
    if action == "retry":
        return {"hitl_status": "pending", "next_agent": "filter_compare_agent", "revision_count": 0,
                "messages": [{"from": "human", "to": AGENT, "content": "Requested retry of filtering/scoring"}]}
    # reject
    return {
        "hitl_status": "rejected",
        "hitl_feedback": decision.get("feedback", ""),
        "next_agent": "end",
        "messages": [{"from": "human", "to": AGENT, "content": f"Rejected: {decision.get('feedback', '')}"}],
    }
