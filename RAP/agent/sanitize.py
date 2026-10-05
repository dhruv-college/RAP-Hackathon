"""Prompt-injection mitigation for untrusted document content (page text, headings, quotes).

Everything from the document is wrapped in a per-session random delimiter, e.g.
<DOC_7f3a9c0b12de> ... </DOC_7f3a9c0b12de>, after stripping sequences that could close the
wrapper or impersonate a role. This is mitigation, not a guarantee.
"""
from __future__ import annotations

import re
import secrets

_ROLE_MARKERS = [
    r"<\|[^|>]{0,40}\|>",                                    # <|im_start|>, <|system|> ...
    r"\[/?(?:INST|SYS)\]",                                   # [INST] [/INST] [SYS]
    r"<<\s*/?\s*SYS\s*>>",
    r"(?im)^\s*#{2,}\s*(?:system|assistant|user|developer|instructions?)\b[^\n]*$",  # ### System
    r"</[^<>\n]{0,60}>",                                     # any closing tag </...>
    r"<\s*(?:system|assistant|user|developer|tool|instructions?)\b[^>]{0,80}>",       # <system ...>
    r"<\s*/?\s*DOC_[0-9a-fA-F]*\s*>",                        # anything mimicking our delimiter
]
_COMPILED = [re.compile(p) for p in _ROLE_MARKERS]


def new_delimiter() -> str:
    return "DOC_" + secrets.token_hex(6)


def clean(text: str, delim: str, max_chars: int | None = None) -> str:
    text = text or ""
    for _ in range(3):  # repeat: one removal can expose another tag-like sequence
        before = text
        for pattern in _COMPILED:
            text = pattern.sub("[removed]", text)
        text = re.sub(re.escape(delim), "[removed]", text, flags=re.IGNORECASE)
        if text == before:
            break
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n[TRUNCATED]"
    return text


def wrap(text: str, delim: str, max_chars: int | None = None) -> str:
    return f"<{delim}>\n{clean(text, delim, max_chars)}\n</{delim}>"
