"""
Shared long-term memory: a JSON-file-backed store of past sessions
(requirement -> recommendation -> feedback trail), separate from the
LangGraph in-run checkpoint state. This is what the "Memory Viewer" in
the UI reads, and what lets the Feedback Agent say "last time you also
cared about battery life" across sessions, not just within one run.
"""
import json
import os
import time
from utils.logger import get_logger

log = get_logger("memory")

_MEM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory_store")
os.makedirs(_MEM_DIR, exist_ok=True)
_MEM_PATH = os.path.join(_MEM_DIR, "sessions.json")


def _load_all() -> dict:
    if not os.path.exists(_MEM_PATH):
        return {}
    try:
        with open(_MEM_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load memory store, starting fresh: {e}")
        return {}


def _save_all(data: dict):
    with open(_MEM_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_turn(session_id: str, turn_record: dict):
    data = _load_all()
    data.setdefault(session_id, []).append({**turn_record, "ts": time.time()})
    _save_all(data)


def get_session_history(session_id: str) -> list:
    return _load_all().get(session_id, [])


def get_all_sessions() -> dict:
    return _load_all()


def clear_all():
    _save_all({})
