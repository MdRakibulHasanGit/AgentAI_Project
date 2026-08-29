"""
Filter & Comparison Agent.

Step 1: hard filter (budget, min RAM, min storage).
Step 2: weighted scoring, computed by handing an explicit formula to
the Python code-execution tool (tools/python_tool.run_python) rather
than hardcoding the math inline -- this is the system's "code
execution" tool-usage agent.

Normalization is against the FULL catalog (tools/vectorstore.get_kb().all()),
not just the surviving shortlist, so a 1-2 candidate shortlist doesn't
get its scores artificially stretched across 0-100 (a bug fixed in the
original single-agent version of this project).
"""
from tools.vectorstore import get_kb
from tools.python_tool import run_python
from utils.logger import TRACE

AGENT = "filter_compare_agent"

BASE_WEIGHTS = {"CPU": 0.30, "RAM": 0.20, "GPU": 0.20, "storage": 0.15, "battery": 0.10, "price": 0.05}
_PRIORITY_MULTIPLIER = {"low": 0.5, "medium": 1.0, "high": 1.5}

_SCORING_SNIPPET = """
def normalize(value, min_v, max_v, higher_is_better=True):
    if max_v == min_v:
        return 100.0
    score = (value - min_v) / (max_v - min_v) * 100
    return score if higher_is_better else (100 - score)

results = []
for p in products:
    breakdown = {
        "CPU": normalize(p["cpu_score"], min(cpu_vals), max(cpu_vals)) * weights["CPU"],
        "RAM": normalize(p["ram_gb"], min(ram_vals), max(ram_vals)) * weights["RAM"],
        "GPU": normalize(p["gpu_score"], min(gpu_vals), max(gpu_vals)) * weights["GPU"],
        "storage": normalize(p["storage_gb"], min(storage_vals), max(storage_vals)) * weights["storage"],
        "battery": normalize(p["battery_hours"], min(battery_vals), max(battery_vals)) * weights["battery"],
        "price": normalize(p["price"], min(price_vals), max(price_vals), higher_is_better=False) * weights["price"],
    }
    total = round(sum(breakdown.values()), 1)
    results.append({"product": p, "score": total, "breakdown": breakdown})
results.sort(key=lambda r: r["score"], reverse=True)
"""


def _hard_filter(products: list, requirement: dict) -> list:
    budget = requirement.get("budget")
    min_ram = requirement.get("min_ram_gb", 0)
    min_storage = requirement.get("min_storage_gb", 0)
    out = []
    for p in products:
        if budget and p["price"] > budget:
            continue
        if p["ram_gb"] < min_ram:
            continue
        if p["storage_gb"] < min_storage:
            continue
        out.append(p)
    return out


def _adjusted_weights(requirement: dict) -> dict:
    weights = dict(BASE_WEIGHTS)
    priorities = requirement.get("priorities", {})
    for key in ("CPU", "RAM", "GPU", "battery"):
        mult = _PRIORITY_MULTIPLIER.get(priorities.get(key, "medium"), 1.0)
        weights[key] *= mult
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    requirement = state.get("requirement", {})
    candidates = state.get("rag_candidates", [])

    survivors = _hard_filter(candidates, requirement)
    TRACE.log("message", AGENT, f"Hard filter: {len(candidates)} -> {len(survivors)} candidates survived")

    if not survivors:
        TRACE.set_status(AGENT, "done")
        TRACE.log("agent_end", AGENT, "No survivors after hard filter")
        return {
            "scored_candidates": [],
            "messages": [{"from": AGENT, "to": "supervisor", "content": "No candidates survived the hard filter"}],
            "next_agent": "recommendation_agent",
        }

    catalog = get_kb().all()
    weights = _adjusted_weights(requirement)

    TRACE.log("tool_call", AGENT, "python_tool.run_python(scoring formula)")
    ns = run_python(_SCORING_SNIPPET, {
        "products": survivors,
        "weights": weights,
        "cpu_vals": [p["cpu_score"] for p in catalog],
        "ram_vals": [p["ram_gb"] for p in catalog],
        "gpu_vals": [p["gpu_score"] for p in catalog],
        "storage_vals": [p["storage_gb"] for p in catalog],
        "battery_vals": [p["battery_hours"] for p in catalog],
        "price_vals": [p["price"] for p in catalog],
    })
    scored = ns["results"]

    TRACE.log("message", AGENT, f"Scored {len(scored)} candidates (normalized against full {len(catalog)}-item catalog)")
    TRACE.set_status(AGENT, "done")
    top = scored[0] if scored else None
    TRACE.log("agent_end", AGENT, f"Top: {top['product']['name'] if top else 'none'} ({top['score'] if top else 0}/100)")

    return {
        "scored_candidates": scored,
        "messages": [{"from": AGENT, "to": "critic_agent", "content": f"Scored {len(scored)} candidates, top={scored[0]['product']['name'] if scored else 'none'}"}],
        "next_agent": "critic_agent",
    }
