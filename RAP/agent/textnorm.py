"""Text normalization shared by search_keyword and the verifier, so quote checks and search agree."""
from __future__ import annotations

import re
import unicodedata

_MAP = {
    "­": "",                                   # soft hyphen
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
}


def normalize(text: str, join_hyphens: bool = True) -> str:
    """Lowercase, NFKC, fold ligatures/quotes/dashes, drop soft hyphens, join words hyphenated
    across a line break ("inter-\\nnational" -> "international"), collapse whitespace.

    join_hyphens=False keeps a hyphen that ends a line ("well-\\nknown" -> "well-known");
    callers check both forms so genuine hyphenated compounds still match.
    """
    if not text:
        return ""
    for src, dst in _MAP.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    if join_hyphens:
        text = re.sub(r"(\w)-[ \t]*\r?\n\s*(\w)", r"\1\2", text)
    else:
        text = re.sub(r"(\w)-[ \t]*\r?\n\s*(\w)", r"\1-\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalized_forms(text: str) -> tuple[str, str]:
    return normalize(text, True), normalize(text, False)


def keyword_variants(keyword: str) -> set[str]:
    """The keyword plus a simple English singular, so 'allowances' also finds 'allowance'.
    (A singular keyword already matches plurals because longer keywords match as substrings.)"""
    kw = normalize(keyword)
    out = {kw} if kw else set()
    words = kw.split(" ")
    last = words[-1] if words else ""
    stem = None
    if len(last) > 4 and last.endswith("ies"):
        stem = last[:-3] + "y"
    elif len(last) > 4 and last.endswith(("ches", "shes", "sses", "xes")):
        stem = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        stem = last[:-1]
    if stem:
        out.add(" ".join(words[:-1] + [stem]))
    return out
