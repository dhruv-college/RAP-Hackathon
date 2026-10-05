"""AgentLoop: one question = understand -> (plan -> gate -> read)* -> answer once -> verify.

run() is a generator of trace events so the UI can stream them. Per-question state lives in
local variables and is discarded when the question ends; nothing page-derived carries over.
"""
from __future__ import annotations

import secrets
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator

from . import answerer, llm, planner, reader
from . import effort as effort_policy
from .config import CONFIG, INSUFFICIENT_MESSAGE
from .document import Document
from .gate import BudgetExhausted, ToolGate
from .planner import PlanState
from .sanitize import new_delimiter
from .schemas import FinalAnswer
from .tools import NO_TEXT_LAYER
from .verify import Evidence, check_final, inside_injection, quote_in_page, repair_quote


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: "s_" + secrets.token_hex(4))
    delim: str = field(default_factory=new_delimiter)
    history: list[str] = field(default_factory=list)  # earlier questions only (intent), never page content


def _preview(result, limit=120) -> str:
    if isinstance(result, list):
        return f"{len(result)} headings" if result and isinstance(result[0], dict) else (f"pages {result}" if result else "no matches")
    text = " ".join(str(result).split())
    return text[:limit] + ("..." if len(text) > limit else "")


class AgentLoop:
    def __init__(self, doc: Document, session: Session | None = None):
        self.doc = doc
        self.session = session or Session()

    def run(self, question: str) -> Iterator[dict]:
        sess = self.session
        qid = "q_" + secrets.token_hex(4)
        ctx = {"session_id": sess.session_id, "question_id": qid, "llm_calls": 0}
        started = time.time()
        gate = ToolGate(self.doc.doc_id, self.doc.n_pages, session_id=sess.session_id, question_id=qid)
        st = PlanState(question=question, title=self.doc.title, n_pages=self.doc.n_pages,
                       delim=sess.delim, history=list(sess.history))
        yield {"type": "start", "question_id": qid, "budget": gate.max_calls}

        # 1. Understand the question first (no tool calls)
        try:
            st.understanding = planner.understand(question, sess.history, self.doc.title, self.doc.n_pages, ctx)
        except llm.LLMError as exc:
            st.understanding = planner.fallback_understanding(question)
            yield {"type": "note", "level": "warning", "text": f"Question analysis failed ({exc}); using keywords from the question."}
        st.missing = list(st.understanding.sub_parts)
        yield {"type": "understand", "understanding": st.understanding.model_dump()}

        # Reasoning effort for this question, chosen from the question analysis
        ctx["effort"], why = effort_policy.choose(st.understanding)
        yield {"type": "effort", "effort": dict(ctx["effort"]), "reason": why}

        # 2. Plan / act
        error = None
        step = 0
        for _ in range(CONFIG.max_planner_steps):
            if gate.exhausted:
                yield {"type": "note", "level": "info", "text": "Tool budget spent; going straight to the answer."}
                break
            try:
                action = planner.plan(st, gate.remaining, gate.used, ctx)
            except llm.LLMError as exc:
                error = str(exc)
                yield {"type": "note", "level": "error", "text": f"Planner unavailable: {exc}"}
                break
            if not CONFIG.use_reader and action.evidence:
                self._absorb_planner_evidence(st, action.evidence)
            if action.missing:
                st.missing = action.missing

            if action.action == "finish":
                cands = planner.unread_candidates(st)
                if not st.evidence and gate.remaining > 0 and cands and not st.nudged:
                    st.nudged = True
                    ctx["effort"], changed = effort_policy.escalate(ctx["effort"], struggling=True)
                    if changed:
                        yield {"type": "effort", "effort": dict(ctx["effort"]), "reason": f"raised: {changed} (no evidence yet)"}
                    st.feedback = (f"You chose finish with no evidence, but pages {cands[:4]} look relevant and are "
                                   "unread. Read the most promising one unless you are sure the document cannot answer.")
                    yield {"type": "note", "level": "info", "text": f"Planner tried to finish with no evidence; asked it to check pages {cands[:4]} first."}
                    continue
                yield {"type": "step", "step": step + 1, "tool": "finish", "args": {}, "status": "ok",
                       "result_preview": "", "used": gate.used, "total": gate.max_calls, "reason": action.reason, "flags": []}
                break

            try:
                res = gate.execute(action)
            except BudgetExhausted:
                break
            step += 1
            flags: list[str] = []
            if res.rejected:
                st.feedback = f"{res.tool}({res.args}) was rejected: {res.error}. Choose something else."
                flags.append(res.status.replace("_", " "))
                yield self._step_event(step, action, res, gate, flags, "")
                continue
            st.feedback = ""
            label = f"{res.tool}({', '.join(f'{k}={v!r}' for k, v in res.args.items())})"
            if not res.ok:
                st.tool_log.append(f"- {label} -> ERROR {res.error}")
                yield self._step_event(step, action, res, gate, ["error"], res.error or "")
                continue

            preview = _preview(res.result)
            if res.tool == "list_headings":
                st.headings = [h for h in res.result if h.get("page", -1) >= 1]
                st.tool_log.append(f"- {label} -> {len(st.headings)} headings")
            elif res.tool == "search_keyword":
                st.searches[res.args["keyword"]] = res.result
                st.tool_log.append(f"- {label} -> {res.result or 'no matches'}")
            elif res.tool == "get_page":
                page = res.args["page"]
                note, flags, preview = self._handle_page(st, page, res.result, question, ctx)
                st.page_notes[page] = note
                st.tool_log.append(f"- {label} -> {note}")
            yield self._step_event(step, action, res, gate, flags, preview)

        # 3. The final answer call: exactly once
        ctx["effort"], changed = effort_policy.escalate(ctx["effort"], evidence_pages=len({e.page for e in st.evidence}))
        if changed:
            yield {"type": "effort", "effort": dict(ctx["effort"]), "reason": f"raised: {changed} (evidence from several pages)"}
        empty_pages = sorted(p for p, n in st.page_notes.items() if not any(e.page == p for e in st.evidence))
        try:
            final = answerer.answer(question, st.understanding, st.evidence, empty_pages,
                                    st.instruction_pages, sess.delim, ctx)
        except llm.LLMError as exc:
            error = error or str(exc)
            final = FinalAnswer(status="insufficient", answer=INSUFFICIENT_MESSAGE, citations=[])
        final, issues = check_final(final, st.evidence, gate.page_texts, st.instruction_spans)
        for issue in issues:
            yield {"type": "note", "level": "info", "text": f"Verifier: {issue}"}

        text = self._render(final, st, error)
        sess.history.append(question)
        yield {"type": "final", "question_id": qid, "status": final.status, "answer": text,
               "citations": [c.model_dump() for c in final.citations], "error": error,
               "stats": {"tool_calls": gate.used, "budget": gate.max_calls, "llm_calls": ctx["llm_calls"],
                         "latency_s": round(time.time() - started, 1), "pages_read": sorted(st.page_notes),
                         "searches": st.searches, "evidence": len(st.evidence),
                         "effort": dict(ctx["effort"])}}

    # ------------------------------------------------------------------ #
    def _handle_page(self, st: PlanState, page: int, text: str, question: str, ctx) -> tuple[str, list[str], str]:
        flags: list[str] = []
        if text == NO_TEXT_LAYER:
            return "no text layer (scanned page?), unreadable", ["no text layer"], NO_TEXT_LAYER
        if not CONFIG.use_reader:
            st.raw_pages[page] = text
            return "read; text shown below for you to quote", flags, _preview(text)
        try:
            r = reader.read_page(question, st.understanding, page, text, st.delim, ctx)
        except llm.LLMError as exc:
            return f"reader failed ({str(exc)[:80]})", ["reader error"], _preview(text)
        st.evidence.extend(r.kept)
        parts = [f"{len(r.kept)} relevant quote(s) kept" if r.kept else "nothing relevant to the question"]
        if r.repaired:
            parts.append(f"{r.repaired} repaired to verbatim text")
        if r.dropped:
            parts.append(f"{r.dropped} unverifiable quote(s) dropped")
        if r.cut_off:
            st.cut_off_pages.append(page)
            parts.append("relevant text continues on the next page")
            flags.append("cut off")
        if r.instruction_like:
            st.instruction_pages.append(page)
            if r.instruction_text:
                st.instruction_spans.append(r.instruction_text)
            parts.append("contains instruction-like text aimed at an AI (ignored)")
            flags.append("instruction-like")
        if r.kept:
            flags.append(f"{len(r.kept)} quote(s)")
        return "; ".join(parts), flags, _preview(text)

    def _absorb_planner_evidence(self, st: PlanState, quotes: list[str]) -> None:
        for quote in quotes:
            for page, text in st.raw_pages.items():
                q = quote if quote_in_page(quote, text) else repair_quote(quote, text)
                if q and not inside_injection(q, st.instruction_spans) and not any(e.quote == q for e in st.evidence):
                    st.evidence.append(Evidence(page=page, quote=q))
                    break

    @staticmethod
    def _step_event(step, action, res, gate, flags, preview) -> dict:
        return {"type": "step", "step": step, "tool": res.tool, "args": res.args, "status": res.status,
                "result_preview": preview, "used": gate.used, "total": gate.max_calls,
                "reason": action.reason, "flags": flags}

    @staticmethod
    def _render(final: FinalAnswer, st: PlanState, error: str | None) -> str:
        if final.status == "answered":
            sources = "\n".join(f"- p.{c.page}: “{unicodedata.normalize('NFKC', c.quote)}”"
                                for c in final.citations)
            return f"{final.answer.strip()}\n\n**Sources**\n{sources}"
        searched = ", ".join(f"\"{k}\"" for k in st.searches) or "none"
        text = f"{INSUFFICIENT_MESSAGE} (Searched: {searched}; pages read: {sorted(st.page_notes) or 'none'}.)"
        if final.answer and final.answer.strip() and final.answer.strip() != INSUFFICIENT_MESSAGE:
            text += f"\n\n{final.answer.strip()}"
        if error:
            text += f"\n\n⚠ LLM error: {error[:300]}"
        return text
