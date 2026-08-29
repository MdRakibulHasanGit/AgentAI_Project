"""
Critic / QA Agent.

Reviews the top-scored candidate against the original requirement and
flags inconsistencies before the Recommendation Agent writes
user-facing text -- e.g. "requirement says gaming + high GPU priority,
but the top pick only has an integrated GPU" (this is precisely the
class of bug found and fixed by hand in the original single-agent
version of this project; here a dedicated agent catches it
automatically as part of the pipeline).

Demonstrates real agent-to-agent collaboration: if the critic rejects,
the Supervisor routes back to the Filter & Comparison Agent for one
revision pass (max 1 retry) instead of silently shipping a flawed
recommendation.
"""
from utils.logger import TRACE

AGENT = "critic_agent"
MAX_REVISIONS = 1


def _is_dedicated_gpu(gpu_name: str) -> bool:
    return "integrated" not in gpu_name.lower()


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    scored = state.get("scored_candidates", [])
    requirement = state.get("requirement", {})
    revision_count = state.get("revision_count", 0)

    issues = []
    if not scored:
        issues.append("No candidates survived filtering -- budget or RAM/storage constraints may be too strict.")
    else:
        top = scored[0]["product"]
        use_cases = requirement.get("use_case", [])
        priorities = requirement.get("priorities", {})

        if ("gaming" in use_cases or "video_editing" in use_cases) and priorities.get("GPU") == "high":
            if not _is_dedicated_gpu(top["gpu"]):
                issues.append(
                    f"Top pick '{top['name']}' has GPU priority=high for {use_cases} but only an integrated GPU ({top['gpu']})."
                )

        if requirement.get("budget") and top["price"] > requirement["budget"] * 1.05:
            issues.append(f"Top pick price ৳{top['price']:,} exceeds stated budget ৳{requirement['budget']:,} by more than 5%.")

    approved = len(issues) == 0 or revision_count >= MAX_REVISIONS
    verdict = {"approved": approved, "issues": issues, "revision_count": revision_count}

    if issues:
        TRACE.log("message", AGENT, f"Found {len(issues)} issue(s): {issues}", level="warn")
    else:
        TRACE.log("message", AGENT, "No issues found, recommendation approved")

    TRACE.set_status(AGENT, "done")

    if not approved:
        TRACE.log("agent_end", AGENT, "Sending back to filter_compare_agent for revision", level="warn")
        return {
            "critic_verdict": verdict,
            "revision_count": revision_count + 1,
            "messages": [{"from": AGENT, "to": "filter_compare_agent", "content": f"Revision requested: {issues}"}],
            "next_agent": "filter_compare_agent",
        }

    TRACE.log("agent_end", AGENT, f"Approved (issues noted for transparency: {len(issues)})")
    return {
        "critic_verdict": verdict,
        "messages": [{"from": AGENT, "to": "recommendation_agent", "content": "Approved, proceed to recommendation"}],
        "next_agent": "recommendation_agent",
    }
