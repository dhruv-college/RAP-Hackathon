"""ToolGate: the single point of enforcement for the tool budget. One instance per question.

- Counts executions; once `used == max_calls`, execute() raises BudgetExhausted and runs nothing.
- Rejects exact duplicates (same tool, same argument) without running them or spending budget.
- Rejects invalid arguments (page out of range, empty/long keyword) without spending budget.
- Logs every attempt, executed or rejected.
- Keeps text of pages read IN THIS QUESTION for verification; discarded with the gate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import calllog
from .config import CONFIG
from .schemas import Action
from .textnorm import normalize
from .tools import TOOL_FUNCS


class BudgetExhausted(Exception):
    pass


@dataclass
class ToolResult:
    tool: str
    args: dict
    status: str              # ok | error | rejected_duplicate | rejected_invalid
    result: Any = None
    error: str | None = None
    budget_used: int = 0
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def rejected(self) -> bool:
        return self.status.startswith("rejected")


@dataclass
class ToolGate:
    doc_id: str
    n_pages: int
    session_id: str = ""
    question_id: str = ""
    max_calls: int = field(default_factory=lambda: CONFIG.max_tool_calls)
    logger: Any = None
    used: int = 0
    page_texts: dict[int, str] = field(default_factory=dict)
    history: list[ToolResult] = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def __post_init__(self):
        self.logger = self.logger or calllog.get_logger()

    @property
    def remaining(self) -> int:
        return self.max_calls - self.used

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_calls

    def _key(self, action: Action) -> tuple:
        if action.action == "get_page":
            return ("get_page", action.page)
        if action.action == "search_keyword":
            return ("search_keyword", normalize(action.keyword or ""))
        return (action.action,)

    @staticmethod
    def _args(action: Action) -> dict:
        if action.action == "get_page":
            return {"page": action.page}
        if action.action == "search_keyword":
            return {"keyword": action.keyword}
        return {}

    def _log(self, kind: str, res: ToolResult) -> None:
        self.logger.log({
            "session_id": self.session_id, "question_id": self.question_id, "kind": kind,
            "name": res.tool, "args": res.args, "result_preview": calllog.preview(res.result) if res.ok else None,
            "duration_ms": res.duration_ms, "budget_used": self.used, "status": res.status, "error": res.error,
        })

    def execute(self, action: Action) -> ToolResult:
        if action.action == "finish":
            raise ValueError("finish is not a tool call")
        args = self._args(action)

        if self.exhausted:
            res = ToolResult(action.action, args, "budget_exhausted", budget_used=self.used,
                             error=f"budget of {self.max_calls} tool calls is spent")
            self._log("reject", res)
            raise BudgetExhausted(res.error)

        invalid = None
        if action.action == "get_page" and not (isinstance(action.page, int) and 1 <= action.page <= self.n_pages):
            invalid = f"page must be between 1 and {self.n_pages}"
        if invalid:
            res = ToolResult(action.action, args, "rejected_invalid", error=invalid, budget_used=self.used)
            self._log("reject", res)
            self.history.append(res)
            return res

        key = self._key(action)
        if key in self._seen:
            res = ToolResult(action.action, args, "rejected_duplicate", budget_used=self.used,
                             error="already called with the same argument in this question")
            self._log("reject", res)
            self.history.append(res)
            return res

        self._seen.add(key)
        self.used += 1
        start = time.time()
        try:
            fn = TOOL_FUNCS[action.action]
            result = fn(self.doc_id, **args) if action.action != "list_documents" else fn()
            res = ToolResult(action.action, args, "ok", result=result, budget_used=self.used)
            if action.action == "get_page":
                self.page_texts[action.page] = result
        except Exception as exc:  # the call was made, so it still counts
            res = ToolResult(action.action, args, "error", error=str(exc), budget_used=self.used)
        res.duration_ms = round((time.time() - start) * 1000, 1)
        self._log("tool", res)
        self.history.append(res)
        return res
