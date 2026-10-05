"""Call logging: every tool attempt and every LLM call is recorded, so nothing is hidden.

Event shape (one JSON object per line in logs/calls.jsonl):
  {"ts", "session_id", "question_id", "kind": "tool|llm|reject", "name", "args",
   "result_preview", "duration_ms", "budget_used", "status", "error", ...}
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

from .config import CONFIG


class CallLogger(Protocol):
    def log(self, event: dict) -> None: ...


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(self, event: dict) -> None:
        event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


class OrganizerWrapperLogger:
    """Pass-through to the organizers' provided call-logging wrapper.

    TODO(wire-organizer-wrapper): when the wrapper is available, forward events here, e.g.
        from organizer_logging import log_event
        log_event(event)
    If their wrapper is a decorator over the tool functions instead, use `organizer_wrap` below.
    """

    def log(self, event: dict) -> None:
        return None


class MultiLogger:
    def __init__(self, loggers: list[CallLogger]):
        self.loggers = loggers

    def log(self, event: dict) -> None:
        for lg in self.loggers:
            lg.log(event)


_logger: CallLogger = MultiLogger([JsonlLogger(CONFIG.log_path), OrganizerWrapperLogger()])


def get_logger() -> CallLogger:
    return _logger


def set_logger(logger: CallLogger) -> None:
    global _logger
    _logger = logger


def organizer_wrap(fn: Callable) -> Callable:
    """TODO(wire-organizer-wrapper): if the organizers' wrapper is a decorator, apply it here:
        from organizer_logging import log_call
        return log_call(fn)
    Every tool in tools.TOOL_FUNCS passes through this function."""
    return fn


def preview(value, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"


def read_events(path: Path | None = None, session_id: str | None = None,
                question_id: str | None = None) -> list[dict]:
    path = Path(path or CONFIG.log_path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ev = json.loads(line)
            if session_id and ev.get("session_id") != session_id:
                continue
            if question_id and ev.get("question_id") != question_id:
                continue
            out.append(ev)
    return out
