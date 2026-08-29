"""
Builds and compiles the LangGraph StateGraph wiring together:
  Supervisor -> {Requirement | Feedback} -> Web Search -> RAG ->
  Filter/Compare <-> Critic -> Recommendation -> Human Review -> Report -> END

A MemorySaver checkpointer makes every run resumable, which is what
powers the human-in-the-loop pause/approve/retry controls in the UI.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from graph.supervisor import run as supervisor_run, route_generic
from graph.human_review import run as human_review_run
from agents.requirement_agent import run as requirement_run
from agents.feedback_agent import run as feedback_run
from agents.websearch_agent import run as websearch_run
from agents.rag_agent import run as rag_run
from agents.filter_compare_agent import run as filter_run
from agents.critic_agent import run as critic_run
from agents.recommendation_agent import run as recommendation_run
from agents.report_agent import run as report_run

NODE_FUNCS = {
    "supervisor": supervisor_run,
    "requirement_agent": requirement_run,
    "feedback_agent": feedback_run,
    "web_search_agent": websearch_run,
    "rag_agent": rag_run,
    "filter_compare_agent": filter_run,
    "critic_agent": critic_run,
    "recommendation_agent": recommendation_run,
    "human_review": human_review_run,
    "report_agent": report_run,
}

# every node (except supervisor, handled separately) can route to any of these
_ROUTABLE = {name: name for name in NODE_FUNCS if name != "supervisor"}
_ROUTABLE["end"] = END


def build_graph():
    graph = StateGraph(AgentState)

    for name, fn in NODE_FUNCS.items():
        graph.add_node(name, fn)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_generic, {
        "requirement_agent": "requirement_agent",
        "feedback_agent": "feedback_agent",
    })

    for name in ["requirement_agent", "feedback_agent", "web_search_agent", "rag_agent",
                 "filter_compare_agent", "critic_agent", "recommendation_agent", "human_review"]:
        graph.add_conditional_edges(name, route_generic, _ROUTABLE)

    graph.add_edge("report_agent", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP
