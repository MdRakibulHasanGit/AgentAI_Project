"""
RAG knowledge base over the laptop catalog.

Uses a TF-IDF vector space (scikit-learn) instead of a hosted embedding
API -- this keeps the whole system runnable offline / without any API
key, while still being genuine vector retrieval (cosine similarity over
a document-term matrix), not keyword matching. Swapping this for a real
embedding model (OpenAI/Anthropic/local sentence-transformers) later is
a one-function change: `_embed()`.
"""
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import get_logger

log = get_logger("tools.vectorstore")

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "laptops.json")


def _use_case_tags(p: dict) -> str:
    """Derives semantic use-case tags from specs so retrieval works on
    *capability* (e.g. "good for gaming") not just literal product-name
    substrings. Without this, a query like "gaming laptop" only matches
    the handful of products whose marketing name happens to contain the
    word "Gaming" -- everything else with a strong dedicated GPU gets
    silently excluded from retrieval."""
    tags = []
    is_dedicated_gpu = "integrated" not in p["gpu"].lower()
    if is_dedicated_gpu and p["gpu_score"] >= 60:
        tags += ["gaming", "game", "dedicated gpu", "video editing", "rendering", "design"]
    elif is_dedicated_gpu:
        tags += ["light gaming", "casual gaming"]
    if p["cpu_score"] >= 70:
        tags += ["programming", "coding", "development", "compiling", "multitasking"]
    if p["ram_gb"] >= 16:
        tags += ["multitasking", "heavy multitasking"]
    if p["battery_hours"] >= 8:
        tags += ["long battery", "all day battery", "portable"]
    if p["weight_kg"] <= 1.8:
        tags += ["lightweight", "portable", "travel"]
    if p["cpu_score"] < 65 and not is_dedicated_gpu:
        tags += ["office", "browsing", "student", "everyday use"]
    return " ".join(tags)


def _doc_text(p: dict) -> str:
    """Flattens a laptop record into a text document for vectorization."""
    return (
        f"{p['name']} {p['brand']} {p['cpu']} {p['gpu']} "
        f"{p['ram_gb']}GB RAM {p['storage_gb']}GB {p['storage_type']} "
        f"{p['screen']} battery {p['battery_hours']}h weight {p['weight_kg']}kg "
        f"price {p['price']} taka {_use_case_tags(p)}"
    )


class LaptopKnowledgeBase:
    """A tiny in-process RAG store: TF-IDF index over data/laptops.json."""

    def __init__(self):
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            self.products: list[dict] = json.load(f)
        self._docs = [_doc_text(p) for p in self.products]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._docs)
        log.info(f"Indexed {len(self.products)} laptops into TF-IDF vector store")

    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        """Semantic-ish retrieval: cosine similarity in TF-IDF space."""
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked_idx = sims.argsort()[::-1][:k]
        return [self.products[i] for i in ranked_idx if sims[i] > 0] or self.products[:k]

    def all(self) -> list[dict]:
        return list(self.products)


_KB: LaptopKnowledgeBase | None = None


def get_kb() -> LaptopKnowledgeBase:
    global _KB
    if _KB is None:
        _KB = LaptopKnowledgeBase()
    return _KB
