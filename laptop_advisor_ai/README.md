# Laptop Advisor AI -- Multi-Agent System (LangGraph + Streamlit)

An AI workforce (not a chatbot) that recommends laptops: a **Supervisor
Agent** coordinates **8 specialized agents** through a LangGraph graph,
with real tool usage, a RAG knowledge base, persistent memory, and a
human-in-the-loop approval step -- all observable through a Streamlit
dashboard.

Runs with **zero API key** (every agent has a rule-based fallback).
Set `ANTHROPIC_API_KEY` in `.env` to switch the reasoning agents over
to real LLM calls -- no code changes needed.

## Quickstart

```bash
cd laptop_advisor_ai
pip install -r requirements.txt
streamlit run app.py
```

Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to
enable LLM-powered reasoning (otherwise runs in free rule-based mode).

## Architecture

```
Supervisor
  ├─ (turn 1)   → Requirement Analysis Agent
  └─ (turn 2+)  → Feedback / Re-planning Agent (preserves prior context)
                          │
                          ▼
                   Web Search Agent  (real tool: DuckDuckGo)
                          │
                          ▼
                   RAG Retrieval Agent  (TF-IDF vector KB over data/laptops.json)
                          │
                          ▼
        ┌──────► Filter & Scoring Agent  (python code-execution tool)
        │                 │
        │                 ▼
        │          Critic / QA Agent ──── rejects? ──► back to Filter & Scoring (max 1 retry)
        │                 │ approves
        │                 ▼
        │          Recommendation Agent
        │                 │
        │                 ▼
        │          Human Review (HITL) ── retry ───────┘
        │                 │ approve
        │                 ▼
        │          Report Writer Agent → END
        └── reject → END
```

## Rubric coverage

| Requirement | Where |
|---|---|
| Supervisor Agent | `graph/supervisor.py` -- makes the turn-1-vs-follow-up routing decision, distinct from worker agents |
| 6+ specialized agents | 8 agents in `agents/`: Requirement, Feedback, Web Search, RAG, Filter/Scoring, Critic, Recommendation, Report Writer |
| Agent-to-agent communication | Every agent appends to `state["messages"]` (`from`/`to`/`content`); Critic → Filter/Scoring back-edge is a real inter-agent request for revision, visible in the "Agent Communication" UI tab |
| Shared memory / knowledge base | `graph/state.py` (in-run blackboard, checkpointed) + `memory/shared_memory.py` (persistent cross-session JSON store) + `tools/vectorstore.py` (RAG knowledge base) |
| Tool integration | Web search (`tools/web_search_tool.py`, DuckDuckGo, no key needed), RAG (`tools/vectorstore.py`, TF-IDF vector retrieval), Python code execution (`tools/python_tool.py`, runs the scoring formula) |
| Planning & reasoning | Requirement Agent extracts structured intent; Filter/Scoring computes weighted multi-criteria scores; Critic reasons about consistency before allowing output |
| Memory | Working memory (LangGraph state, checkpointed via `MemorySaver` for pause/resume) + long-term memory (`memory/shared_memory.py`) + knowledge base (vector store) |
| Human-in-the-loop | `graph/human_review.py` uses LangGraph's native `interrupt()`; UI has Approve / Retry / Reject controls that resume the exact paused run via `Command(resume=...)` |
| Logging & error handling | `utils/logger.py` -- file + console logging, in-memory `TraceRecorder` for the UI, `TRACE.log_exception()` helper; every tool degrades gracefully instead of crashing (e.g. web search returns `[]` if offline) |
| Interactive UI (Streamlit) | `app.py` -- 6 tabs: Chat & Recommendation, Live Execution Trace, Agent Communication, Execution Graph (Mermaid, from `app.get_graph().draw_mermaid()`), Memory Viewer, Logs & Errors; sidebar has live agent-status dashboard + token/cost panel |
| Token usage & cost estimation | `utils/token_tracker.py`, reported by `utils/llm_client.py` on every real LLM call, rendered in the sidebar |

## Project layout

```
laptop_advisor_ai/
├── app.py                     # Streamlit UI (entry point)
├── graph/
│   ├── state.py                # shared LangGraph state (blackboard)
│   ├── supervisor.py           # Supervisor Agent + routing helpers
│   ├── human_review.py         # HITL interrupt() node
│   └── build_graph.py          # StateGraph wiring + checkpointer
├── agents/                     # 8 specialized agents (see table above)
├── tools/
│   ├── web_search_tool.py      # real web search (DuckDuckGo)
│   ├── vectorstore.py          # RAG knowledge base (TF-IDF)
│   └── python_tool.py          # sandboxed code-execution tool
├── memory/shared_memory.py     # persistent cross-session memory
├── utils/
│   ├── logger.py                # logging + live trace recorder
│   ├── token_tracker.py         # token/cost estimator
│   ├── llm_client.py            # Anthropic wrapper (optional)
│   └── models.py                # Requirement schema + rule-based parser
├── data/laptops.json           # 20-laptop demo catalog
├── requirements.txt
└── .env.example
```

## Notes for the demo

- **No API key needed to run or present this.** Every agent's
  rule-based fallback is the real code path being exercised, not a
  stub -- worth saying explicitly during presentation.
- The sandbox this was built in has restricted outbound network
  access, so the Web Search Agent may return zero results in that
  specific environment; it will hit the live internet normally when
  run on your own machine or Streamlit Cloud. The rest of the
  pipeline is unaffected either way (that's the point of the
  graceful-degradation design in `tools/web_search_tool.py`).
- To watch the Critic Agent actually reject and retry live, try a
  request where the budget only allows integrated-GPU laptops but the
  use case is "gaming" -- e.g. lower the budget until the catalog's
  cheapest dedicated-GPU laptop no longer qualifies.
