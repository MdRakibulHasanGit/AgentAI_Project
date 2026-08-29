"""Data shapes shared across agents."""
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class Requirement:
    category: str = "laptop"
    budget: Optional[int] = None
    use_case: List[str] = field(default_factory=list)
    priorities: Dict[str, str] = field(default_factory=lambda: {
        "CPU": "medium", "RAM": "medium", "GPU": "medium", "battery": "medium",
    })
    min_ram_gb: int = 0
    min_storage_gb: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Requirement":
        return Requirement(
            category=d.get("category", "laptop"),
            budget=d.get("budget"),
            use_case=d.get("use_case", []),
            priorities=d.get("priorities", {"CPU": "medium", "RAM": "medium", "GPU": "medium", "battery": "medium"}),
            min_ram_gb=d.get("min_ram_gb", 0),
            min_storage_gb=d.get("min_storage_gb", 0),
        )


_BUDGET_PATTERNS = [
    r"(\d[\d,]*)\s*(?:হাজার|k)\b",
    r"[৳$]?\s*(\d[\d,]{3,})",
]

_USE_CASE_KEYWORDS = {
    "gaming": ["gaming", "game", "গেমিং", "গেম"],
    "programming": ["programming", "coding", "developer", "প্রোগ্রামিং", "কোডিং"],
    "video_editing": ["video editing", "editing", "premiere", "video edit"],
    "office": ["office", "excel", "word", "browsing", "study", "student"],
    "design": ["design", "photoshop", "illustrator", "graphics"],
}


def parse_requirement_rule_based(text: str, previous: Optional[Requirement] = None) -> Requirement:
    """Very small heuristic parser -- good enough for the demo's rule-based
    fallback path when no LLM key is configured."""
    req = previous or Requirement()
    t = text.lower()

    for pat in _BUDGET_PATTERNS:
        m = re.search(pat, t)
        if m:
            raw = m.group(1).replace(",", "")
            val = int(raw)
            if "হাজার" in t or re.search(r"\d\s*k\b", t):
                val *= 1000
            req.budget = val
            break

    for use_case, keywords in _USE_CASE_KEYWORDS.items():
        if any(k in t for k in keywords) and use_case not in req.use_case:
            req.use_case.append(use_case)

    if "gaming" in req.use_case:
        req.priorities["GPU"] = "high"
        req.priorities["CPU"] = "high"
    if "programming" in req.use_case:
        req.priorities["CPU"] = "high"
        req.priorities["RAM"] = "high"
    if "video_editing" in req.use_case or "design" in req.use_case:
        req.priorities["GPU"] = "high"
        req.priorities["RAM"] = "high"

    if any(w in t for w in ["battery", "ব্যাটারি", "light", "halka", "হালকা", "weight"]):
        req.priorities["battery"] = "high"
    if any(w in t for w in ["16gb", "16 gb"]):
        req.min_ram_gb = max(req.min_ram_gb, 16)
    if any(w in t for w in ["8gb", "8 gb"]):
        req.min_ram_gb = max(req.min_ram_gb, 8)

    if not req.use_case:
        req.use_case = ["general"]

    return req
