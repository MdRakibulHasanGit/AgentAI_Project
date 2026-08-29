"""
Python code-execution tool.

Runs a small, restricted snippet of python against a supplied namespace
and returns the resulting variables. Used by the Filter & Scoring Agent
to execute the weighted-scoring formula as an explicit "tool call"
(rather than hardcoding it as inline agent logic), which is what the
assignment means by "code execution" tool usage: the agent decides the
weights/formula, the tool runs the math.

This is a demo-grade sandbox (restricted builtins, no imports, no file
or network access) -- adequate for running arithmetic on trusted
in-process data, not a general-purpose untrusted-code sandbox.
"""
from utils.logger import get_logger

log = get_logger("tools.python_exec")

_SAFE_BUILTINS = {
    "min": min, "max": max, "sum": sum, "len": len, "round": round,
    "sorted": sorted, "abs": abs, "range": range, "enumerate": enumerate,
}


def run_python(code: str, local_vars: dict) -> dict:
    """Executes `code` with `local_vars` pre-populated. Returns the
    resulting local namespace (minus builtins). Raises on error -- the
    caller is expected to catch and log via TRACE."""
    ns = dict(local_vars)
    exec(code, {"__builtins__": _SAFE_BUILTINS}, ns)
    for k in list(local_vars.keys()):
        ns.pop(k, None)
    log.info(f"Executed python tool snippet, produced vars: {list(ns.keys())}")
    return ns
