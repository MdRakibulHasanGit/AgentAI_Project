"""
Report Writer Agent.

Runs after human-in-the-loop approval. Compiles everything the pipeline
produced (requirement, top pick, reasons, alternatives, critic notes,
web search context) into one polished Markdown report for the final
output viewer / download button in the UI.
"""
from utils.logger import TRACE

AGENT = "report_agent"


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    req = state.get("requirement", {})
    rec = state.get("recommendation", {})
    web_findings = state.get("web_findings", [])
    critic = state.get("critic_verdict", {})

    lines = ["# Laptop recommendation report", ""]
    lines.append(f"**Budget:** ৳{req.get('budget'):,}" if req.get("budget") else "**Budget:** not specified")
    lines.append(f"**Use case:** {', '.join(req.get('use_case', []))}")
    lines.append(f"**Priorities:** {req.get('priorities', {})}")
    lines.append("")

    top = rec.get("top_pick")
    if top:
        p = top["product"]
        lines.append(f"## 🥇 Recommended: {p['name']} -- ৳{p['price']:,} (score {top['score']}/100)")
        lines.append(f"- CPU: {p['cpu']}")
        lines.append(f"- GPU: {p['gpu']}")
        lines.append(f"- RAM: {p['ram_gb']}GB | Storage: {p['storage_gb']}GB {p['storage_type']}")
        lines.append(f"- Battery: ~{p['battery_hours']}h | Weight: {p['weight_kg']}kg")
        lines.append("")
        lines.append("### Why this pick")
        for r in rec.get("reasons", []):
            lines.append(f"- {r}")
        lines.append("")
        alts = rec.get("alternatives", {})
        if alts:
            lines.append("### Alternatives")
            for label, sp in alts.items():
                ap = sp["product"]
                lines.append(f"- **{label}**: {ap['name']} -- ৳{ap['price']:,} ({sp['score']}/100)")
            lines.append("")
    else:
        lines.append("## No viable recommendation")
        lines.append(rec.get("message", ""))
        lines.append("")

    if critic.get("issues"):
        lines.append("### QA notes (from Critic Agent)")
        for issue in critic["issues"]:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")

    if web_findings:
        lines.append("### Web context")
        for f in web_findings:
            lines.append(f"- [{f['title']}]({f['url']}) -- {f['snippet'][:140]}")
        lines.append("")

    lines.append("### Assistant message")
    lines.append(rec.get("message", ""))

    report = "\n".join(lines)

    TRACE.set_status(AGENT, "done")
    TRACE.log("agent_end", AGENT, "Final report compiled")

    return {
        "report_markdown": report,
        "messages": [{"from": AGENT, "to": "supervisor", "content": "Report compiled and ready"}],
        "next_agent": "end",
    }
