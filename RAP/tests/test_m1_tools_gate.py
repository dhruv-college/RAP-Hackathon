import pytest

from agent import tools
from agent.gate import BudgetExhausted, ToolGate
from agent.schemas import Action
from agent.textnorm import keyword_variants, normalize


# ---------------- textnorm ----------------
def test_normalize_ligatures_quotes_whitespace():
    assert normalize("Ofﬁce  “workﬂow”\n\tTAX") == 'office "workflow" tax'


def test_normalize_soft_hyphen_and_linebreak_hyphenation():
    assert normalize("re­imburse") == "reimburse"
    assert normalize("inter-\nnational") == "international"
    assert normalize("well-\nknown", join_hyphens=False) == "well-known"


def test_keyword_variants_plural_folding():
    assert "allowance" in keyword_variants("Allowances")
    assert "policy" in keyword_variants("policies")
    assert keyword_variants("class") == {"class"}


# ---------------- document + tools ----------------
def test_page_indexing_is_one_based(doc_outline):
    assert "Welcome to ACME" in tools.get_page(doc_outline.doc_id, 1)
    assert "45 days" in tools.get_page(doc_outline.doc_id, doc_outline.n_pages)
    for bad in (0, doc_outline.n_pages + 1, -1):
        with pytest.raises(ValueError):
            tools.get_page(doc_outline.doc_id, bad)


def test_blank_page_returns_no_text_layer(doc_outline):
    assert tools.get_page(doc_outline.doc_id, 4) == tools.NO_TEXT_LAYER


def test_search_returns_page_numbers_only(doc_outline):
    hits = tools.search_keyword(doc_outline.doc_id, "NOTICE PERIOD")
    assert hits == [2, 5]
    assert all(type(p) is int for p in hits)
    assert tools.search_keyword(doc_outline.doc_id, "allowance") == [3]   # plural on the page
    assert tools.search_keyword(doc_outline.doc_id, "Allowances") == [3]
    assert tools.search_keyword(doc_outline.doc_id, "xyzzy") == []


def test_search_short_keyword_matches_whole_word(doc_outline):
    assert tools.search_keyword(doc_outline.doc_id, "tax") == [3]
    assert tools.search_keyword(doc_outline.doc_id, "ax") == []


def test_list_headings_toc_path(doc_outline):
    heads = tools.list_headings(doc_outline.doc_id)
    assert heads == [{"title": "Introduction", "page": 1}, {"title": "Leave policy", "page": 2},
                     {"title": "Benefits", "page": 3}, {"title": "Addendum", "page": 5}]


def test_list_headings_fallback_path(doc_plain):
    heads = tools.list_headings(doc_plain.doc_id)
    titles = [h["title"] for h in heads]
    assert {"Introduction", "Leave policy", "Benefits", "Addendum"} <= set(titles)
    assert all(set(h) == {"title", "page"} for h in heads)
    assert "ACME Handbook - confidential" not in titles   # running header dropped
    assert not any(t.startswith("Page ") for t in titles)


def test_list_headings_truncation_flag(doc_outline, monkeypatch):
    from agent.config import CONFIG
    monkeypatch.setattr(CONFIG, "max_headings", 2)
    doc_outline.headings_cache = None
    heads = tools.list_headings(doc_outline.doc_id)
    assert len(heads) == 3 and heads[-1] == {"title": "[truncated]", "page": -1}


def test_no_text_extracted_at_upload(doc_outline):
    assert doc_outline.headings_cache is None
    assert not hasattr(doc_outline, "pages")


# ---------------- gate ----------------
def test_gate_allows_six_and_refuses_seventh(doc_outline, capture_log):
    gate = ToolGate(doc_outline.doc_id, doc_outline.n_pages, max_calls=6)
    actions = [Action(action="list_headings")] + [Action(action="get_page", page=p) for p in (1, 2, 3, 5)] \
        + [Action(action="search_keyword", keyword="notice")]
    for a in actions:
        assert gate.execute(a).ok
    assert gate.used == 6 and gate.exhausted and gate.remaining == 0
    with pytest.raises(BudgetExhausted):
        gate.execute(Action(action="search_keyword", keyword="leave"))
    assert gate.used == 6
    executed = [e for e in capture_log.events if e["kind"] == "tool"]
    assert len(executed) == 6
    assert capture_log.events[-1]["status"] == "budget_exhausted"


def test_gate_duplicate_page_rejected_without_budget(doc_outline, capture_log):
    gate = ToolGate(doc_outline.doc_id, doc_outline.n_pages)
    assert gate.execute(Action(action="get_page", page=2)).ok
    res = gate.execute(Action(action="get_page", page=2))
    assert res.status == "rejected_duplicate" and gate.used == 1
    assert capture_log.events[-1]["status"] == "rejected_duplicate"


def test_gate_invalid_page_rejected_without_budget(doc_outline):
    gate = ToolGate(doc_outline.doc_id, doc_outline.n_pages)
    res = gate.execute(Action(action="get_page", page=99))
    assert res.status == "rejected_invalid" and gate.used == 0


def test_action_validation():
    with pytest.raises(ValueError):
        Action(action="search_keyword", keyword="  ")
    with pytest.raises(ValueError):
        Action(action="search_keyword", keyword="x" * 101)
    with pytest.raises(ValueError):
        Action(action="get_page")


def test_gate_keeps_page_text_per_question_only(doc_outline):
    gate = ToolGate(doc_outline.doc_id, doc_outline.n_pages)
    gate.execute(Action(action="get_page", page=2))
    assert 2 in gate.page_texts
    assert ToolGate(doc_outline.doc_id, doc_outline.n_pages).page_texts == {}


def test_search_matches_across_hyphenated_line_break():
    import pymupdf
    from agent.document import Document
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Travel claims need reim-\nbursement forms signed by a well-\nknown manager.", fontsize=11)
    d = Document(pdf.tobytes(), "hy.pdf")
    tools.register(d)
    try:
        assert tools.search_keyword(d.doc_id, "reimbursement") == [1]
        assert tools.search_keyword(d.doc_id, "well-known") == [1]
    finally:
        tools.unregister(d.doc_id)
