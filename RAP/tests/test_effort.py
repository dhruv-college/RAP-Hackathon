from agent import effort, llm
from agent.config import CONFIG
from agent.loop import AgentLoop, Session
from agent.schemas import Understanding


def und(qtype, change=False, parts=1):
    return Understanding(restated_question="q", question_type=qtype, answer_shape="", sub_parts=["p"] * parts,
                         search_terms=["x"], likely_sections=[], change_risk=change)


def test_simple_fact_is_cheap_and_change_questions_think_harder():
    simple, _ = effort.choose(und("locate_fact"))
    change, _ = effort.choose(und("comparison_or_change", change=True))
    assert simple == {"planner": "low", "reader": "low", "answerer": "medium"}
    assert change == {"planner": "high", "reader": "low", "answerer": "high"}


def test_modifiers_raise_but_never_lower():
    e, why = effort.choose(und("locate_fact", change=True, parts=3))
    assert e["planner"] == "medium" and e["answerer"] == "high" and "3 sub-parts" in why


def test_ceiling_caps_everything(monkeypatch):
    monkeypatch.setattr(CONFIG, "reasoning_effort_ceiling", "medium")
    e, _ = effort.choose(und("comparison_or_change"))
    assert set(e.values()) <= {"low", "medium"}
    e2, _ = effort.escalate(e, evidence_pages=3, struggling=True)
    assert e2["planner"] == "medium" and e2["answerer"] == "medium"


def test_escalation_mid_question():
    e, _ = effort.choose(und("locate_fact"))
    e2, changed = effort.escalate(e, evidence_pages=2, struggling=True)
    assert e2["planner"] == "high" and e2["answerer"] == "high" and "planner low->high" in changed


def test_adaptive_off_uses_fixed_values(monkeypatch):
    monkeypatch.setattr(CONFIG, "adaptive_effort", False)
    e, why = effort.choose(und("comparison_or_change"))
    assert e == effort.fixed() and "fixed" in why


def test_chosen_effort_reaches_the_llm_calls(doc_outline, capture_log):
    def fake(role, messages, mode, payload):
        if role == "understand":
            return {"restated_question": "q", "question_type": "comparison_or_change", "answer_shape": "",
                    "sub_parts": ["p"], "search_terms": ["notice"], "likely_sections": [], "change_risk": True}
        if role == "planner":
            return {"name": "finish", "arguments": {"reason": "r", "missing": []}}
        return {"status": "insufficient", "answer": "", "citations": []}
    llm.set_fake(fake)
    events = list(AgentLoop(doc_outline, Session()).run("What is the notice period now?"))
    by_role = {e["name"]: e["reasoning_effort"] for e in capture_log.events if e["kind"] == "llm"}
    assert by_role["understand"] == CONFIG.reasoning_effort_understand
    assert by_role["planner"] == "high" and by_role["answerer"] == "high"
    assert any(e["type"] == "effort" for e in events)
    assert events[-1]["stats"]["effort"]["planner"] == "high"
