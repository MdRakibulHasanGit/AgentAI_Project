"""
Token usage + API cost estimator.

Every LLM call in the system (see utils/llm_client.py) reports its
input/output token counts here. When no API key is configured, agents
run in rule-based fallback mode and simply never call `record()`, so
usage stays at zero -- the panel still renders, it just shows the
system is running for free.

Prices are USD per 1M tokens, illustrative rates for the Claude family.
Update PRICE_TABLE if you point this at a different model/provider.
"""
import threading

PRICE_TABLE = {
    # model_name: (input $/1M tokens, output $/1M tokens)
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-opus-4-8": (15.00, 75.00),
    "default": (3.00, 15.00),
}


class TokenTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._calls = []  # list of dicts: agent, model, input_tokens, output_tokens, cost

    def reset(self):
        with self._lock:
            self._calls = []

    def record(self, agent: str, model: str, input_tokens: int, output_tokens: int):
        in_price, out_price = PRICE_TABLE.get(model, PRICE_TABLE["default"])
        cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
        with self._lock:
            self._calls.append({
                "agent": agent,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            })

    def summary(self):
        with self._lock:
            calls = list(self._calls)
        total_in = sum(c["input_tokens"] for c in calls)
        total_out = sum(c["output_tokens"] for c in calls)
        total_cost = sum(c["cost_usd"] for c in calls)
        return {
            "calls": calls,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_usd": round(total_cost, 6),
        }


TOKENS = TokenTracker()
