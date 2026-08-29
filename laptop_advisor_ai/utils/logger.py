"""
Central logging + execution trace recorder.

Two things live here:
1. A standard python `logging` setup that writes to logs/run.log (for
   real observability / debugging).
2. `TRACE`, an in-memory event recorder that the Streamlit UI polls to
   render the live agent execution trace, communication history, and
   error panel. This is intentionally a plain list behind a lock --
   no external dependency needed for a single-process demo.
"""
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass
class TraceEvent:
    ts: float
    kind: str          # "agent_start" | "agent_end" | "message" | "tool_call" | "error" | "hitl"
    agent: str
    detail: str
    payload: Optional[dict] = None
    level: str = "info"  # info | warn | error


class TraceRecorder:
    """Thread-safe append-only trace log shared between the LangGraph run
    (background thread) and the Streamlit UI (main thread)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events: list[TraceEvent] = []
        self._agent_status: dict[str, str] = {}  # agent -> idle|running|done|error

    def clear(self):
        with self._lock:
            self._events = []
            self._agent_status = {}

    def log(self, kind: str, agent: str, detail: str, payload: dict = None, level: str = "info"):
        ev = TraceEvent(ts=time.time(), kind=kind, agent=agent, detail=detail, payload=payload, level=level)
        with self._lock:
            self._events.append(ev)
        logger = get_logger(f"agent.{agent}")
        msg = f"[{kind}] {detail}"
        if level == "error":
            logger.error(msg)
        elif level == "warn":
            logger.warning(msg)
        else:
            logger.info(msg)

    def set_status(self, agent: str, status: str):
        with self._lock:
            self._agent_status[agent] = status

    def snapshot(self):
        with self._lock:
            return list(self._events), dict(self._agent_status)

    def log_exception(self, agent: str, exc: Exception):
        tb = traceback.format_exc()
        self.log("error", agent, f"{type(exc).__name__}: {exc}", payload={"traceback": tb}, level="error")
        self.set_status(agent, "error")


TRACE = TraceRecorder()
