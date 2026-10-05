"""The four document tools: pure functions over registered Documents.

These are the ONLY path from the agent to document content. The agent never calls them
directly; it goes through ToolGate (gate.py), which counts, validates and logs every call.
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter

from . import calllog
from .config import CONFIG
from .document import Document
from .textnorm import keyword_variants, normalized_forms

NO_TEXT_LAYER = "[NO_TEXT_LAYER]"
TRUNCATED = "[TRUNCATED]"

_REGISTRY: dict[str, Document] = {}


def register(doc: Document) -> str:
    _REGISTRY[doc.doc_id] = doc
    return doc.doc_id


def unregister(doc_id: str) -> None:
    _REGISTRY.pop(doc_id, None)


def get_doc(doc_id: str) -> Document:
    if doc_id not in _REGISTRY:
        raise KeyError(f"unknown doc_id {doc_id}")
    return _REGISTRY[doc_id]


# --------------------------------------------------------------------------- #
def list_documents() -> list[dict]:
    """Titles and metadata only."""
    return [{"doc_id": d.doc_id, "title": d.title, "pages": d.n_pages, "metadata": dict(d.metadata)}
            for d in _REGISTRY.values()]


def list_headings(doc_id: str) -> list[dict]:
    """[{title, page}] from the PDF outline, or a font-based fallback. Nothing else."""
    doc = get_doc(doc_id)
    if doc.headings_cache is None:
        heads = _outline(doc) or _font_headings(doc)
        if len(heads) > CONFIG.max_headings:
            heads = heads[:CONFIG.max_headings] + [{"title": "[truncated]", "page": -1}]
        doc.headings_cache = heads
    return [dict(h) for h in doc.headings_cache]


def get_page(doc_id: str, page: int) -> str:
    """Text of exactly one page (1-based)."""
    doc = get_doc(doc_id)
    text = doc.page_text(page)
    if len(re.sub(r"\s", "", text)) < 20:
        return NO_TEXT_LAYER
    if len(text) > CONFIG.max_page_chars:
        text = text[:CONFIG.max_page_chars] + "\n" + TRUNCATED
    return text


def search_keyword(doc_id: str, keyword: str) -> list[int]:
    """Sorted 1-based page numbers where the keyword appears. Page numbers only.

    Case-insensitive; tolerant of ligatures, curly quotes, soft hyphens and words hyphenated
    across line breaks; simple plurals folded. Keywords of 3 characters or fewer must match
    a whole word. Page text is extracted on demand and not kept after the call.
    """
    doc = get_doc(doc_id)
    variants = [v for v in keyword_variants(keyword) if v]
    if not variants:
        return []
    patterns = [re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)") if len(v) <= 3 else None for v in variants]
    hits = []
    for page in range(1, doc.n_pages + 1):
        forms = normalized_forms(doc.page_text(page))
        if any((p.search(f) if p else v in f) for v, p in zip(variants, patterns) for f in forms):
            hits.append(page)
    return hits


# --------------------------------------------------------------------------- #
def _clean_title(title: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", title).split())


def _outline(doc: Document) -> list[dict]:
    out = []
    for _level, title, page in doc.toc():
        title = _clean_title(title)
        if title and page >= 1:
            out.append({"title": title, "page": int(page)})
    return out


def _font_headings(doc: Document) -> list[dict]:
    """Heading if font size >= 1.2x the median body size, or bold, <= 80 chars and not ending
    with a period. Running headers/footers and page-number lines are dropped."""
    lines = []  # (page, text, size, bold)
    for page in range(1, doc.n_pages + 1):
        for text, size, bold in doc.page_lines(page):
            lines.append((page, _clean_title(text), size, bold))
    if not lines:
        return []
    sizes = [size for _, text, size, _ in lines for _ in range(max(1, len(text) // 20))]
    median = statistics.median(sizes)

    pages_with = Counter()
    for text in {(p, t) for p, t, _, _ in lines}:
        pages_with[text[1]] += 1
    repeated = {t for t, n in pages_with.items() if n >= 3 and n >= 0.4 * doc.n_pages}

    out, seen = [], set()
    for page, text, size, bold in lines:
        if len(text) < 3 or text in repeated:
            continue
        if re.fullmatch(r"[\d\W]+", text) or re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", text, re.I):
            continue
        is_heading = size >= 1.2 * median or (bold and len(text) <= 80 and not text.endswith("."))
        if is_heading and (page, text) not in seen:
            seen.add((page, text))
            out.append({"title": text, "page": page})
    return out


TOOL_FUNCS = {
    "list_documents": calllog.organizer_wrap(list_documents),
    "list_headings": calllog.organizer_wrap(list_headings),
    "get_page": calllog.organizer_wrap(get_page),
    "search_keyword": calllog.organizer_wrap(search_keyword),
}
