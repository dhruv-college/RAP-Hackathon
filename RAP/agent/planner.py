"""Question understanding (before any tool call) and the planner (one native tool call per step).

The planner never sees raw page text when USE_READER=true: only headings, search hit lists,
verified quotes and short status notes, with all document-derived text wrapped as untrusted data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import llm, prompts
from .config import CONFIG
from .sanitize import wrap
from .schemas import Action, Understanding, planner_tools
from .verify import Evidence

_STOP = set("""a an the of to in on for and or is are was were be been by with as at from what which who whom whose
when where why how does do did can could should would will shall may might must this that these those it its
there their they them i we you your our my me about into than then any all each per under over between
document according mention mentioned state stated say says tell give list please""".split())


@dataclass
class PlanState:
    question: str
    title: str
    n_pages: int
    delim: str
    history: list[str] = field(default_factory=list)
    understanding: Understanding | None = None
    headings: list[dict] | None = None
    searches: dict[str, list[int]] = field(default_factory=dict)
    page_notes: dict[int, str] = field(default_factory=dict)     # page -> short status for the planner
    evidence: list[Evidence] = field(default_factory=list)
    raw_pages: dict[int, str] = field(default_factory=dict)      # only when USE_READER=false
    instruction_spans: list[str] = field(default_factory=list)
    instruction_pages: list[int] = field(default_factory=list)
    cut_off_pages: list[int] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    tool_log: list[str] = field(default_factory=list)
    feedback: str = ""
    nudged: bool = False


# --------------------------------------------------------------------------- #
# 1. Understand the question (no tool calls)
# --------------------------------------------------------------------------- #
def fallback_understanding(question: str) -> Understanding:
    words = [w for w in re.findall(r"[A-Za-z0-9][\w\-%.,]*[\w%]|\w", question) if w.lower() not in _STOP]
    words = sorted(dict.fromkeys(words), key=len, reverse=True)
    return Understanding(restated_question=question, question_type="locate_fact", answer_shape="",
                         sub_parts=[question], search_terms=words[:5], likely_sections=words[:3], change_risk=True)


def understand(question: str, history: list[str], title: str, n_pages: int, ctx=None) -> Understanding:
    earlier = "\n".join(f"- {h[:200]}" for h in history[-3:]) or "(none)"
    user = (f"QUESTION: {question}\nEARLIER QUESTIONS IN THIS CHAT:\n{earlier}\n"
            f"DOCUMENT: \"{title}\", {n_pages} pages (nothing has been read yet)")
    und = llm.call_llm("understand", [{"role": "system", "content": prompts.UNDERSTAND_SYSTEM},
                                      {"role": "user", "content": user}], Understanding, ctx=ctx)
    und.search_terms = [t.strip() for t in und.search_terms if t and t.strip()][:6]
    und.sub_parts = [s for s in und.sub_parts if s and s.strip()][:4] or [question]
    return und


# --------------------------------------------------------------------------- #
# 2. Planner
# --------------------------------------------------------------------------- #
def _analysis_block(u: Understanding) -> str:
    parts = "\n".join(f"  {i}. {p}" for i, p in enumerate(u.sub_parts, 1))
    return (f"- restated: {u.restated_question}\n- type: {u.question_type}\n- a complete answer needs: {u.answer_shape}\n"
            f"- sub-parts:\n{parts}\n- search terms (document wording): {', '.join(u.search_terms)}\n"
            f"- likely section headings: {', '.join(u.likely_sections)}\n"
            f"- value may be amended later in the document: {'yes' if u.change_risk else 'no'}")


def unread_candidates(st: PlanState) -> list[int]:
    """Pages that look relevant but have not been read: search hits (first and last first),
    then pages whose heading shares a word with the analysis terms."""
    read = set(st.page_notes)
    out: list[int] = []
    for hits in st.searches.values():
        ordered = ([hits[0], hits[-1]] if hits else []) + list(hits)
        for p in ordered:
            if p not in read and p not in out:
                out.append(p)
    if st.headings and st.understanding:
        terms = {w.lower() for t in st.understanding.search_terms + st.understanding.likely_sections
                 for w in re.findall(r"\w{4,}", t)}
        for h in st.headings:
            title_words = {w.lower() for w in re.findall(r"\w{4,}", h["title"])}
            if h["page"] >= 1 and terms & title_words and h["page"] not in read and h["page"] not in out:
                out.append(h["page"])
    return out


def build_planner_messages(st: PlanState, remaining: int, used: int) -> list[dict]:
    system = prompts.PLANNER_SYSTEM.format(remaining=remaining, max_calls=CONFIG.max_tool_calls, delim=st.delim)
    earlier = "\n".join(f"- {h[:200]}" for h in st.history[-3:]) or "(none)"
    if st.headings is None:
        heads = "(not fetched yet)"
    elif not st.headings:
        heads = "(the document has no detectable headings; use search_keyword)"
    else:
        heads = wrap("\n".join(f"p{h['page']}: {h['title']}" for h in st.headings), st.delim)
    searches = "\n".join(f"- \"{k}\" -> {v if v else 'no matches'}" for k, v in st.searches.items()) or "(none)"
    notes = "\n".join(f"- p{p}: {n}" for p, n in sorted(st.page_notes.items())) or "(none)"
    if st.evidence:
        ev = wrap("\n".join(f"[p{e.page}] \"{e.quote}\"" for e in sorted(st.evidence, key=lambda e: e.page)), st.delim)
    else:
        ev = "(none yet)"
    cands = unread_candidates(st)
    user = f"""QUESTION: {st.question}
EARLIER QUESTIONS (intent only): {earlier}

QUESTION ANALYSIS:
{_analysis_block(st.understanding)}

DOCUMENT: "{st.title}", {st.n_pages} pages
BUDGET: {used} used, {remaining} of {CONFIG.max_tool_calls} left

TOOL CALLS SO FAR:
{chr(10).join(st.tool_log) or '(none)'}

HEADINGS:
{heads}

SEARCH RESULTS:
{searches}

PAGES READ:
{notes}

VERIFIED EVIDENCE SO FAR:
{ev}

UNREAD PAGES THAT LOOK RELEVANT: {cands[:8] or 'none'}
STILL MISSING (last estimate): {st.missing or 'nothing recorded yet'}"""
    if not CONFIG.use_reader and st.raw_pages:
        raw = "\n\n".join(f"PAGE {p}:\n{wrap(t, st.delim, CONFIG.max_page_chars)}" for p, t in sorted(st.raw_pages.items()))
        user += f"\n\nPAGE TEXT (untrusted data; copy verbatim quotes into `evidence`):\n{raw}"
    if st.feedback:
        user += f"\n\nCONTROLLER NOTE: {st.feedback}"
    user += "\n\nCall exactly one tool."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _to_action(name: str | None, args: dict, text: str) -> Action:
    if name is None:
        # some models answer in text; accept a JSON action if one is there
        data = llm.extract_json(text)
        name, args = data.get("action") or data.get("tool"), data
    if "__invalid_json__" in args:
        raise ValueError("tool arguments were not valid JSON")
    page = args.get("page")
    if isinstance(page, str) and page.strip().isdigit():
        page = int(page.strip())
    return Action(action=name, keyword=args.get("keyword"), page=page, reason=str(args.get("reason", ""))[:300],
                  missing=[str(m) for m in (args.get("missing") or [])][:6],
                  evidence=[str(e) for e in (args.get("evidence") or [])][:6])


def plan(st: PlanState, remaining: int, used: int, ctx=None) -> Action:
    messages = build_planner_messages(st, remaining, used)
    tools = planner_tools(CONFIG.use_reader)
    for attempt in range(2):
        name, args, text = llm.call_tools("planner", messages, tools, ctx=ctx)
        try:
            action = _to_action(name, args, text)
            if st.understanding:
                action.question_type = st.understanding.question_type
            return action
        except (ValidationError, ValueError, TypeError) as exc:
            messages = messages + [{"role": "user", "content":
                f"Your last choice was invalid ({str(exc)[:300]}). Call exactly one of the tools "
                "list_headings, search_keyword, get_page or finish, with valid arguments."}]
    return Action(action="finish", reason="planner output invalid after one retry")
