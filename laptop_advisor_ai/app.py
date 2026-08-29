"""
Streamlit UI for the Laptop Advisor multi-agent system.

Panels: agent dashboard, live execution trace, agent communication
history, execution graph, token/cost estimator, logs & errors, memory
viewer, human-in-the-loop controls, final report viewer.
"""
import sys
import os
import uuid
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.build_graph import get_app, NODE_FUNCS
from langgraph.types import Command
from utils.logger import TRACE, LOG_PATH
from utils.token_tracker import TOKENS
from memory import shared_memory

st.set_page_config(page_title="Laptop Advisor -- Multi-Agent AI", layout="wide")

AGENT_ORDER = [
    "supervisor", "requirement_agent", "feedback_agent", "web_search_agent",
    "rag_agent", "filter_compare_agent", "critic_agent", "recommendation_agent",
    "human_review", "report_agent",
]
AGENT_LABELS = {
    "supervisor": "🧭 Supervisor",
    "requirement_agent": "🧠 Requirement Analysis",
    "feedback_agent": "🔄 Feedback / Re-planning",
    "web_search_agent": "🌐 Web Search",
    "rag_agent": "📚 RAG Retrieval",
    "filter_compare_agent": "⚖️ Filter & Scoring",
    "critic_agent": "🔍 Critic / QA",
    "recommendation_agent": "🏆 Recommendation",
    "human_review": "🧑 Human Review (HITL)",
    "report_agent": "📝 Report Writer",
}

# ---------- session bootstrapping ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "turn" not in st.session_state:
    st.session_state.turn = 0

app = get_app()
config = {"configurable": {"thread_id": st.session_state.thread_id}}


def run_graph(payload):
    """Invokes the graph and stores result + interrupt (if any) in session_state."""
    result = app.invoke(payload, config=config)
    st.session_state.graph_state = result
    interrupts = result.get("__interrupt__")
    st.session_state.pending_interrupt = interrupts[0].value if interrupts else None


# ================= SIDEBAR =================
with st.sidebar:
    st.title("💻 Laptop Advisor AI")
    st.caption("Multi-agent workforce -- Supervisor + 8 specialized agents")

    st.subheader("Agent dashboard")
    events, status_map = TRACE.snapshot()
    for name in AGENT_ORDER:
        status = status_map.get(name, "idle")
        icon = {"idle": "⚪", "running": "🟡", "done": "🟢", "error": "🔴"}.get(status, "⚪")
        st.markdown(f"{icon} {AGENT_LABELS[name]}  \n`{status}`")

    st.divider()
    st.subheader("💰 Token usage & cost")
    summary = TOKENS.summary()
    c1, c2 = st.columns(2)
    c1.metric("Total tokens", summary["total_tokens"])
    c2.metric("Est. cost (USD)", f"${summary['total_cost_usd']:.4f}")
    if summary["calls"]:
        st.dataframe(summary["calls"], use_container_width=True, hide_index=True)
    else:
        st.caption("No LLM calls yet -- running in rule-based fallback mode (no ANTHROPIC_API_KEY set), so cost is $0.")

    st.divider()
    if st.button("🔁 Reset session", use_container_width=True):
        TRACE.clear()
        TOKENS.reset()
        st.session_state.clear()
        st.rerun()

# ================= MAIN =================
tab_chat, tab_trace, tab_comm, tab_graph, tab_memory, tab_logs = st.tabs(
    ["💬 Chat & Recommendation", "🔬 Live Execution Trace", "📨 Agent Communication",
     "🗺️ Execution Graph", "🧠 Memory Viewer", "📋 Logs & Errors"]
)

# ---------- TAB: Chat ----------
with tab_chat:
    st.subheader("Tell the system what laptop you need")
    user_text = st.text_area(
        "Message (Bangla / English / Banglish)",
        placeholder="আমার ৮০ হাজার টাকার মধ্যে programming আর gaming-এর জন্য একটা laptop দরকার",
        height=80,
    )
    run_col, _ = st.columns([1, 4])
    if run_col.button("▶️ Run pipeline", type="primary", disabled=not user_text.strip()):
        with st.spinner("Running multi-agent pipeline..."):
            payload = {"user_text": user_text, "turn": st.session_state.turn, "requirement": (st.session_state.graph_state or {}).get("requirement", {})}
            run_graph(payload)
            st.session_state.turn = st.session_state.graph_state.get("turn", st.session_state.turn)
        st.rerun()

    # Human-in-the-loop controls
    if st.session_state.pending_interrupt:
        st.divider()
        st.warning("⏸️ Pipeline paused -- human approval required")
        pending = st.session_state.pending_interrupt
        st.markdown(f"**Proposed pick:** {pending.get('top_pick') or 'No viable laptop found'}")
        if pending.get("score") is not None:
            st.markdown(f"**Score:** {pending['score']}/100")
        st.markdown(f"**Message:** {pending.get('message', '')}")

        a1, a2, a3 = st.columns(3)
        if a1.button("✅ Approve", use_container_width=True):
            with st.spinner("Resuming pipeline..."):
                run_graph(Command(resume={"action": "approve"}))
            st.rerun()
        if a2.button("🔁 Retry scoring", use_container_width=True):
            with st.spinner("Retrying filter & scoring..."):
                run_graph(Command(resume={"action": "retry"}))
            st.rerun()
        reject_reason = a3.text_input("Rejection reason (optional)", key="reject_reason", label_visibility="collapsed", placeholder="Why reject?")
        if a3.button("❌ Reject", use_container_width=True):
            with st.spinner("Rejecting..."):
                run_graph(Command(resume={"action": "reject", "feedback": reject_reason}))
            st.rerun()

    # Report viewer
    state = st.session_state.graph_state
    if state and state.get("report_markdown"):
        st.divider()
        st.subheader("📄 Final report")
        st.markdown(state["report_markdown"])
        st.download_button("⬇️ Download report (.md)", state["report_markdown"], file_name="laptop_recommendation.md")

    if state and state.get("hitl_status") == "rejected":
        st.error(f"Recommendation rejected. Feedback: {state.get('hitl_feedback', '(none)')}")

# ---------- TAB: Live trace ----------
with tab_trace:
    st.subheader("Live agent execution trace")
    events, _ = TRACE.snapshot()
    if not events:
        st.caption("No events yet -- run the pipeline from the Chat tab.")
    for ev in reversed(events):
        icon = {"agent_start": "▶️", "agent_end": "⏹️", "message": "💬", "tool_call": "🛠️", "error": "🔴", "hitl": "🧑"}.get(ev.kind, "•")
        level_color = {"warn": "orange", "error": "red"}.get(ev.level, "gray")
        st.markdown(f"{icon} **{AGENT_LABELS.get(ev.agent, ev.agent)}** -- :{level_color}[{ev.detail}]")

# ---------- TAB: Agent communication ----------
with tab_comm:
    st.subheader("Agent-to-agent communication history")
    state = st.session_state.graph_state
    msgs = state.get("messages", []) if state else []
    if not msgs:
        st.caption("No messages yet.")
    for m in msgs:
        st.markdown(f"`{m['from']}` → `{m['to']}`  \n{m['content']}")
        st.divider()

# ---------- TAB: Execution graph ----------
with tab_graph:
    st.subheader("LangGraph execution graph")
    try:
        mermaid_src = app.get_graph().draw_mermaid()
        st.markdown(f"```mermaid\n{mermaid_src}\n```")
    except Exception as e:
        st.error(f"Could not render graph: {e}")
    st.caption("Solid edges = fixed flow. Dashed edges = conditional routing decided at runtime by each agent's `next_agent`.")

# ---------- TAB: Memory viewer ----------
with tab_memory:
    st.subheader("Shared memory / knowledge base")
    st.markdown("**Current run's working memory (blackboard state)**")
    state = st.session_state.graph_state
    if state:
        st.json({k: v for k, v in state.items() if k != "messages"})
    else:
        st.caption("No active run.")

    st.markdown("**Long-term session memory (persisted across turns/sessions)**")
    all_sessions = shared_memory.get_all_sessions()
    if all_sessions:
        st.json(all_sessions)
    else:
        st.caption("No persisted sessions yet.")

    from tools.vectorstore import get_kb
    st.markdown(f"**RAG knowledge base** -- {len(get_kb().all())} laptops indexed (TF-IDF vector store)")
    st.dataframe(get_kb().all(), use_container_width=True, hide_index=True)

# ---------- TAB: Logs & errors ----------
with tab_logs:
    st.subheader("Execution logs")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log_content = f.read()[-8000:]
        st.code(log_content, language="log")
    else:
        st.caption("No logs yet.")

    st.subheader("Errors")
    events, _ = TRACE.snapshot()
    errors = [e for e in events if e.level == "error"]
    if errors:
        for e in errors:
            st.error(f"**{e.agent}**: {e.detail}")
            if e.payload and "traceback" in e.payload:
                st.code(e.payload["traceback"])
    else:
        st.success("No errors recorded.")
