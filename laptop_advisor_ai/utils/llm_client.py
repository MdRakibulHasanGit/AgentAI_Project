"""
Thin Anthropic wrapper shared by every agent.

Design goal: the whole system must run with ZERO api key (every agent
has a rule-based fallback), but the moment ANTHROPIC_API_KEY is set,
agents switch to real LLM reasoning automatically -- no code changes.
Every real call reports tokens to utils.token_tracker.TOKENS so the UI
cost panel is accurate.
"""
import os
from utils.token_tracker import TOKENS

_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
_client = None


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def ask_text(agent_name: str, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    """Returns plain text, or raises if the API call fails (caller should
    catch and fall back to rule-based logic)."""
    client = _get_client()
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    usage = getattr(resp, "usage", None)
    if usage:
        TOKENS.record(agent_name, _MODEL, usage.input_tokens, usage.output_tokens)
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip()
