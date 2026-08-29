"""
Shared state (the "blackboard") that flows through the LangGraph graph.
Every agent node reads from and writes back into this single TypedDict --
this is the system's short-term / working memory, checkpointed by
LangGraph's MemorySaver so a run can be paused (human-in-the-loop) and
resumed later without losing context.
"""
from typing import Annotated, Any, Optional, TypedDict
import operator


def _last_wins(a, b):
    return b if b is not None else a


class AgentState(TypedDict, total=False):
    # --- conversation ---
    user_text: str                       # latest raw user input
    session_id: str
    turn: int

    # --- requirement analysis agent output ---
    requirement: dict

    # --- web search agent output (real tool) ---
    web_findings: list                   # list[{query, snippet, url}]

    # --- RAG retrieval agent output ---
    rag_candidates: list                 # list[dict] laptop records retrieved from KB

    # --- filter & compare agent output ---
    scored_candidates: list              # list[{product, score, breakdown}]

    # --- critic agent output ---
    critic_verdict: dict                 # {approved: bool, issues: [...], notes: str}
    revision_count: int

    # --- recommendation agent output ---
    recommendation: dict

    # --- report writer agent output ---
    report_markdown: str

    # --- human-in-the-loop ---
    hitl_status: str                     # "pending" | "approved" | "rejected" | "n/a"
    hitl_feedback: str

    # --- inter-agent communication log (for UI "agent communication history") ---
    messages: Annotated[list, operator.add]

    # --- control flow ---
    next_agent: str
    error: Optional[str]
