import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import calllog, llm, tools  # noqa: E402
from agent.document import Document  # noqa: E402


class ListLogger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


@pytest.fixture(autouse=True)
def capture_log():
    logger = ListLogger()
    calllog.set_logger(logger)
    yield logger
    llm.set_fake(None)


def _pdf(pages: list[str], outline: bool, headings: list[tuple[int, str]] | None = None) -> bytes:
    doc = pymupdf.open()
    for i, body in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)
        if body:
            page.insert_htmlbox(pymupdf.Rect(60, 60, 535, 780), body,
                                css="body{font-family:sans-serif;font-size:11pt} h1{font-size:18pt}")
            page.insert_text((60, 40), "ACME Handbook - confidential", fontsize=8)  # running header
            page.insert_text((280, 815), f"Page {i}", fontsize=8)
    if outline and headings:
        doc.set_toc([[1, t, p] for p, t in headings])
    doc.set_metadata({"title": "ACME Handbook"})
    data = doc.tobytes()
    doc.close()
    return data


PAGES = [
    "<h1>Introduction</h1><p>Welcome to ACME. This handbook explains our policies for all staff members.</p>",
    "<h1>Leave policy</h1><p>Employees receive 20 days of annual leave. The notice period is 30 days for all staff "
    "members who resign voluntarily after probation. The workflow is reviewed every year.</p>",
    "<h1>Benefits</h1><p>Medical insurance covers employees and dependants. Cafeteria allowances are paid "
    "monthly. Our tax policy is described elsewhere.</p>",
    "",  # blank page -> [NO_TEXT_LAYER]
    "<h1>Addendum</h1><p>This addendum replaces the notice period in the Leave policy: the notice period is "
    "now 45 days.</p>",
]
HEADINGS = [(1, "Introduction"), (2, "Leave policy"), (3, "Benefits"), (5, "Addendum")]


@pytest.fixture
def doc_outline():
    d = Document(_pdf(PAGES, True, HEADINGS), "acme.pdf")
    tools.register(d)
    yield d
    tools.unregister(d.doc_id)


@pytest.fixture
def doc_plain():
    d = Document(_pdf(PAGES, False), "acme_plain.pdf")
    tools.register(d)
    yield d
    tools.unregister(d.doc_id)
