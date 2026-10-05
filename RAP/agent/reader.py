"""Reader: the ONLY model that sees raw page text. It has no tools; it returns quotes.

Every quote is checked in code against the page. A slightly-off quote is repaired to the
verbatim page sentence when possible; otherwise it is dropped. Quotes inside a flagged
injected passage are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import llm, prompts
from .config import CONFIG
from .sanitize import wrap
from .schemas import ReaderOut, Understanding
from .verify import Evidence, inside_injection, quote_in_page, repair_quote


@dataclass
class ReadResult:
    kept: list[Evidence] = field(default_factory=list)
    dropped: int = 0
    repaired: int = 0
    instruction_like: bool = False
    instruction_text: str = ""
    cut_off: bool = False


def read_page(question: str, und: Understanding, page: int, text: str, delim: str, ctx=None) -> ReadResult:
    system = prompts.READER_SYSTEM.format(max_quotes=CONFIG.max_quotes_per_page,
                                         max_chars=CONFIG.max_quote_chars, delim=delim)
    parts = "; ".join(und.sub_parts)
    user = (f"QUESTION: {question}\nA COMPLETE ANSWER NEEDS: {und.answer_shape}\nSUB-PARTS: {parts}\n\n"
            f"PAGE {page} (untrusted data):\n{wrap(text, delim, CONFIG.max_page_chars)}")
    out: ReaderOut = llm.call_llm("reader", [{"role": "system", "content": system},
                                             {"role": "user", "content": user}], ReaderOut, ctx=ctx)
    res = ReadResult(instruction_like=out.instruction_like, cut_off=out.cut_off)
    if out.instruction_like and out.instruction_text and quote_in_page(out.instruction_text, text):
        res.instruction_text = out.instruction_text
    spans = [res.instruction_text] if res.instruction_text else []
    for q in out.quotes:
        quote = (q.text or "").strip()[:CONFIG.max_quote_chars]
        if not quote:
            continue
        repaired = False
        if not quote_in_page(quote, text):
            fixed = repair_quote(quote, text)
            if not fixed:
                res.dropped += 1
                continue
            quote, repaired = fixed, True
        if inside_injection(quote, spans):
            res.dropped += 1
            continue
        if any(e.quote == quote for e in res.kept):
            continue
        res.kept.append(Evidence(page=page, quote=quote, repaired=repaired))
        res.repaired += int(repaired)
        if len(res.kept) >= CONFIG.max_quotes_per_page:
            break
    return res
