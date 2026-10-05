"""Pure-code verification. No LLM here.

- quote_in_page: a quote counts only if, after normalization, it is a substring of the page.
- repair_quote: when a reader quote is slightly off (dropped word, changed punctuation), snap it
  to the page sentence that contains all its numbers and most of its words, so evidence stays
  verbatim instead of being thrown away (a main cause of "no answer").
- check_final: every citation must be verifiable against a page read in this question and must
  not come from text flagged as an injected instruction. Answered-without-citations becomes
  insufficient.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import CONFIG, INSUFFICIENT_MESSAGE
from .schemas import Citation, FinalAnswer
from .textnorm import normalize, normalized_forms


@dataclass
class Evidence:
    page: int
    quote: str
    repaired: bool = False


def _q(quote: str) -> str:
    return normalize(quote).strip(" .,;:\"'")


def quote_in_page(quote: str, page_text: str) -> bool:
    q = _q(quote)
    if len(q) < 4:
        return False
    return any(q in form for form in normalized_forms(page_text))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", normalize(text))


def repair_quote(quote: str, page_text: str, min_overlap: float = 0.8) -> str | None:
    qt = _tokens(quote)
    if len(qt) < 3:
        return None
    numbers = {t for t in qt if any(ch.isdigit() for ch in t)}
    joined = re.sub(r"(\w)-[ \t]*\r?\n\s*(\w)", r"\1\2", page_text or "")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", joined) if s.strip()]
    windows = sentences + [a + " " + b for a, b in zip(sentences, sentences[1:])]
    candidates = []
    for w in windows:
        # A window can start with a running header or heading on earlier lines; also try it
        # starting at each later line so the shortest verbatim span wins.
        lines = w.split("\n")
        for i in range(len(lines)):
            candidates.append(" ".join(" ".join(lines[i:]).split()))
    best, best_score = None, 0.0
    for cand in candidates:
        ct = set(_tokens(cand))
        if not numbers <= ct:
            continue
        score = sum(1 for t in qt if t in ct) / len(qt)
        if score > best_score or (score == best_score and best is not None and len(cand) < len(best)):
            best, best_score = cand, score
    if best is None or best_score < min_overlap:
        return None
    best = best[:CONFIG.max_quote_chars]
    return best if quote_in_page(best, page_text) else None


def inside_injection(quote: str, instruction_spans: list[str]) -> bool:
    q = _q(quote)
    if not q:
        return False
    for span in instruction_spans:
        s = _q(span)
        if s and (q in s or (len(s) >= 20 and s in q)):
            return True
    return False


def check_final(final: FinalAnswer, evidence: list[Evidence], page_texts: dict[int, str],
                instruction_spans: list[str] | None = None) -> tuple[FinalAnswer, list[str]]:
    issues: list[str] = []
    spans = instruction_spans or []
    if final.status == "insufficient":
        return FinalAnswer(status="insufficient", answer=final.answer, citations=[]), issues

    valid: list[Citation] = []
    for c in final.citations:
        if inside_injection(c.quote, spans):
            issues.append(f"p.{c.page}: citation is from flagged injected text; dropped")
            continue
        pages = ([c.page] if c.page in page_texts else []) + [p for p in page_texts if p != c.page]
        placed = None
        for p in pages:
            in_evidence = any(e.page == p and _q(c.quote) and _q(c.quote) in _q(e.quote) for e in evidence)
            if in_evidence or quote_in_page(c.quote, page_texts[p]):
                placed = Citation(page=p, quote=c.quote)
                break
        if placed is None and c.page in page_texts:
            fixed = repair_quote(c.quote, page_texts[c.page])
            if fixed and not inside_injection(fixed, spans):
                placed = Citation(page=c.page, quote=fixed)
                issues.append(f"p.{c.page}: citation repaired to the verbatim sentence")
        if placed is None:
            issues.append(f"p.{c.page}: citation not found in any page read; dropped")
            continue
        if placed.page != c.page:
            issues.append(f"citation moved from p.{c.page} to p.{placed.page}, where the quote appears")
        valid.append(placed)

    if not valid:
        issues.append("answered without any verifiable citation; downgraded to insufficient")
        return FinalAnswer(status="insufficient", answer=INSUFFICIENT_MESSAGE, citations=[]), issues
    return FinalAnswer(status="answered", answer=final.answer, citations=valid), issues
