"""
Recommendation Agent.

Turns the critic-approved ranked list into a final, human-readable
recommendation: top pick with honest reasons tied to the requirement,
plus alternatives (budget/performance/battery). GPU claims are gated on
whether the GPU is actually dedicated -- carries forward the fix made
to the original single-agent project.
"""
from utils import llm_client
from utils.logger import TRACE

AGENT = "recommendation_agent"

SYSTEM_PROMPT = """You are a Recommendation Agent for a laptop shopping assistant.
Given the requirement and the top scored laptop, write a short, friendly final
recommendation in Bangla (Banglish is fine) explaining WHY it fits, under 100 words.
Do not invent specs not present in the data."""


def _is_dedicated_gpu(gpu_name: str) -> bool:
    return "integrated" not in gpu_name.lower()


def _build_reasons(top: dict, requirement: dict) -> list:
    p = top["product"]
    use_cases = requirement.get("use_case", [])
    priorities = requirement.get("priorities", {})
    reasons = []

    if "programming" in use_cases:
        reasons.append(f"Strong CPU ({p['cpu']}) for programming/compiling")
    if "gaming" in use_cases or "video_editing" in use_cases:
        if _is_dedicated_gpu(p["gpu"]):
            reasons.append(f"Dedicated GPU ({p['gpu']}) for gaming/rendering")
        else:
            reasons.append(f"Integrated GPU ({p['gpu']}) -- fine for light use, not for demanding gaming")
    reasons.append(f"{p['ram_gb']}GB RAM for smooth multitasking")
    reasons.append(f"{p['storage_gb']}GB {p['storage_type']} storage")
    if priorities.get("battery") == "high":
        reasons.append(f"{p['battery_hours']}+ hour battery for all-day use")
    if requirement.get("budget"):
        reasons.append(f"Within budget (৳{p['price']:,} vs ৳{requirement['budget']:,})")
    return reasons


def _find_alternatives(scored: list) -> dict:
    if not scored:
        return {}
    top_id = scored[0]["product"]["id"]
    best_budget = min(scored, key=lambda r: r["product"]["price"])
    best_perf = max(scored, key=lambda r: r["breakdown"]["CPU"] + r["breakdown"]["GPU"])
    best_battery = max(scored, key=lambda r: r["product"]["battery_hours"])
    alts = {}
    for label, sp in [("Best Budget", best_budget), ("Best Performance", best_perf), ("Best Battery", best_battery)]:
        if sp["product"]["id"] != top_id:
            alts[label] = sp
    return alts


def run(state: dict) -> dict:
    TRACE.set_status(AGENT, "running")
    scored = state.get("scored_candidates", [])
    requirement = state.get("requirement", {})

    if not scored:
        rec = {
            "top_pick": None,
            "reasons": [],
            "alternatives": {},
            "message": "কোনো ল্যাপটপ আপনার budget ও requirement-এর সাথে মিলছে না। Budget বাড়ানোর কথা ভাবতে পারেন।",
        }
        TRACE.set_status(AGENT, "done")
        TRACE.log("agent_end", AGENT, "No recommendation possible")
        return {
            "recommendation": rec,
            "messages": [{"from": AGENT, "to": "supervisor", "content": "No viable recommendation"}],
            "next_agent": "human_review",
        }

    top = scored[0]
    reasons = _build_reasons(top, requirement)
    alternatives = _find_alternatives(scored)

    message = None
    if llm_client.is_available():
        try:
            user_prompt = f"Requirement: {requirement}\nTop pick: {top['product']}\nScore: {top['score']}/100\nReasons: {reasons}"
            message = llm_client.ask_text(AGENT, SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            TRACE.log("message", AGENT, f"LLM message generation failed, using template: {e}", level="warn")

    if not message:
        use_case_str = ", ".join(requirement.get("use_case", []))
        message = f"🥇 Recommended: {top['product']['name']} (Score: {top['score']}/100)\nআপনার {use_case_str} প্রয়োজনের সাথে এই ল্যাপটপটি সবচেয়ে ভালো মিলছে।"

    recommendation = {"top_pick": top, "reasons": reasons, "alternatives": alternatives, "message": message}

    TRACE.set_status(AGENT, "done")
    TRACE.log("agent_end", AGENT, f"Recommended: {top['product']['name']} ({top['score']}/100)")

    return {
        "recommendation": recommendation,
        "messages": [{"from": AGENT, "to": "human_review", "content": f"Proposed: {top['product']['name']} ({top['score']}/100)"}],
        "next_agent": "human_review",
        "hitl_status": "pending",
    }
