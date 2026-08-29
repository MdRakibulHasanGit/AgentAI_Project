"""
Real web search tool (DuckDuckGo via the `ddgs` package -- no API key
required). Used by the Web Search Agent to pull live context (e.g.
general price trends, "is X laptop good for gaming" opinions) that the
static local catalog can't provide.

Network access note: this sandbox's outbound egress is restricted to a
package-registry allowlist, so live calls will raise here -- that's
expected and handled with a graceful fallback so the rest of the
pipeline still runs. In a normal deployment (or Streamlit Cloud) this
hits the real internet.
"""
from utils.logger import get_logger

log = get_logger("tools.web_search")


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Returns a list of {title, snippet, url}. Never raises -- returns
    an empty list with a logged warning on failure, so a network-restricted
    environment degrades gracefully instead of crashing the graph."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
        return results
    except Exception as e:
        log.warning(f"web_search failed for '{query}': {e}")
        return []
