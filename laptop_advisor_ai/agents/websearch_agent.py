"""
Web Search Agent.

Real tool-usage agent: queries the live web (via tools/web_search_tool)
for context the static local catalog can't provide -- e.g. general
buying advice or recent price trends for the user's use case. Findings
are attached to state and later cited by the Report Writer Agent.

Degrades gracefully to an empty result set if the sandbox has no
outbound internet access -- the rest of the pipeline is unaffected.
"""
from tools.web_search_tool import web_search
from utils.logger import TRACE

AGENT = "web_search_agent"


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    req = state.get("requirement", {})
    use_cases = ", ".join(req.get("use_case", [])) or "laptop"
    query = f"best laptop for {use_cases} 2026 buying guide"
    TRACE.log("tool_call", AGENT, f"web_search(query={query!r})")

    findings = web_search(query, max_results=3)

    if findings:
        TRACE.log("message", AGENT, f"Found {len(findings)} web results", payload={"findings": findings})
    else:
        TRACE.log("message", AGENT, "No web results (offline sandbox or no matches) -- continuing without live context", level="warn")

    TRACE.set_status(AGENT, "done")
    TRACE.log("agent_end", AGENT, f"web_findings count={len(findings)}")

    return {
        "web_findings": findings,
        "messages": [{"from": AGENT, "to": "supervisor", "content": f"Retrieved {len(findings)} web results for context"}],
        "next_agent": "rag_agent",
    }
