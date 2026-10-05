"""In-memory PDF document. Opened from bytes at upload; NO text is extracted at upload.

Text is extracted only inside get_page and search_keyword (see tools.py). Page numbers are
1-based everywhere outside this file; conversion to PyMuPDF's 0-based index happens only here.
"""
from __future__ import annotations

import secrets
import threading
from pathlib import Path

import pymupdf


class Document:
    def __init__(self, data: bytes, filename: str = "document.pdf"):
        self.doc_id = "doc_" + secrets.token_hex(4)
        self.filename = Path(filename).name
        self._pdf = pymupdf.open(stream=data, filetype="pdf")
        self._lock = threading.Lock()  # PyMuPDF documents are not thread-safe
        self.n_pages = self._pdf.page_count
        meta = self._pdf.metadata or {}
        self.title = (meta.get("title") or "").strip() or Path(filename).stem
        self.metadata = {k: v for k, v in meta.items()
                         if v and k in ("author", "subject", "creator", "producer", "creationDate", "modDate")}
        self.headings_cache: list[dict] | None = None  # memoized by list_headings (call still counted)

    @classmethod
    def from_path(cls, path: str | Path) -> "Document":
        path = Path(path)
        return cls(path.read_bytes(), path.name)

    def _check(self, page: int) -> int:
        if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= self.n_pages:
            raise ValueError(f"page must be an integer between 1 and {self.n_pages}")
        return page - 1  # 0-based for PyMuPDF

    def page_text(self, page: int) -> str:
        """Text of one page in reading order (blocks sorted top-to-bottom, left-to-right)."""
        idx = self._check(page)
        with self._lock:
            blocks = self._pdf[idx].get_text("blocks", sort=True)
        return "\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())

    def toc(self) -> list[list]:
        with self._lock:
            return self._pdf.get_toc(simple=True)

    def page_lines(self, page: int) -> list[tuple[str, float, bool]]:
        """(text, max font size, all-bold) per visual line, for the heading fallback."""
        idx = self._check(page)
        with self._lock:
            data = self._pdf[idx].get_text("dict")
        out = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = " ".join("".join(s["text"] for s in spans).split())
                size = max(s["size"] for s in spans)
                bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
                out.append((text, size, bold))
        return out
