"""
RAG Retrieval Agent.

Queries the TF-IDF vector knowledge base (tools/vectorstore.py) for
candidate laptops matching the requirement's use case, then does a
cheap budget pre-filter (10% headroom, same spirit as the original
project's search agent) so downstream agents get a manageable
shortlist instead of the whole catalog.
"""
from tools.vectorstore import get_kb
from utils.logger import TRACE

AGENT = "rag_agent"


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    req = state.get("requirement", {})
    use_cases = req.get("use_case", ["general"])
    query = " ".join(use_cases) + " laptop"

    TRACE.log("tool_call", AGENT, f"vectorstore.retrieve(query={query!r})")
    kb = get_kb()
    candidates = kb.retrieve(query, k=16)

    budget = req.get("budget")
    if budget:
        ceiling = budget * 1.10
        candidates = [c for c in candidates if c["price"] <= ceiling]

    TRACE.log("message", AGENT, f"Retrieved {len(candidates)} candidates from knowledge base after budget pre-filter")
    TRACE.set_status(AGENT, "done")
    TRACE.log("agent_end", AGENT, f"rag_candidates count={len(candidates)}")

    return {
        "rag_candidates": candidates,
        "messages": [{"from": AGENT, "to": "supervisor", "content": f"Retrieved {len(candidates)} candidates via RAG"}],
        "next_agent": "filter_compare_agent",
    }
