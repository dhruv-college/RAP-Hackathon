"""Adaptive reasoning effort: pick how hard each role thinks, per question.

The understand step classifies the question before any tool call. Simple fact lookups get
cheap settings; questions about changed values or facts spread over pages get more thinking
where it pays off. Effort is raised again mid-question when the evidence turns out harder
than expected. REASONING_EFFORT_CEILING caps everything (set it to "medium" if the demo is slow).

With ADAPTIVE_EFFORT=false, the fixed REASONING_EFFORT_* values are used instead.
The understand step itself always uses REASONING_EFFORT_UNDERSTAND: it is what does the classifying.
"""
from __future__ import annotations

from .config import CONFIG
from .schemas import Understanding

LEVELS = ["low", "medium", "high"]

# question_type -> effort per role
PROFILES: dict[str, dict[str, str]] = {
    # one fact in one place: finding it is easy, stating it is easy
    "locate_fact":          {"planner": "low",    "reader": "low", "answerer": "medium"},
    # facts spread over pages: navigation matters, and the answer must combine them
    "multi_page":           {"planner": "medium", "reader": "low", "answerer": "high"},
    # a value that may have been amended: must find the later text and resolve the conflict
    "comparison_or_change": {"planner": "high",   "reader": "low", "answerer": "high"},
    # probably absent: search properly before declining, then decline plainly
    "likely_unanswerable":  {"planner": "medium", "reader": "low", "answerer": "medium"},
}


def _cap(level: str) -> str:
    ceiling = CONFIG.reasoning_effort_ceiling if CONFIG.reasoning_effort_ceiling in LEVELS else "high"
    return LEVELS[min(LEVELS.index(level), LEVELS.index(ceiling))]


def at_least(level: str, floor: str) -> str:
    return _cap(LEVELS[max(LEVELS.index(level), LEVELS.index(floor))])


def fixed() -> dict[str, str]:
    return {"planner": CONFIG.reasoning_effort_planner, "reader": CONFIG.reasoning_effort_reader,
            "answerer": CONFIG.reasoning_effort_answerer}


def choose(und: Understanding) -> tuple[dict[str, str], str]:
    """Initial effort per role for this question, and a one-line reason for the trace."""
    if not CONFIG.adaptive_effort:
        return fixed(), "fixed (ADAPTIVE_EFFORT=false)"
    plan = dict(PROFILES.get(und.question_type, PROFILES["locate_fact"]))
    why = [und.question_type]
    if und.change_risk:                      # the value may be amended later: look for the amendment
        plan["planner"] = at_least(plan["planner"], "medium")
        why.append("value may be amended")
    if len(und.sub_parts) >= 3:              # several facts to collect and combine
        plan["planner"] = at_least(plan["planner"], "medium")
        plan["answerer"] = at_least(plan["answerer"], "high")
        why.append(f"{len(und.sub_parts)} sub-parts")
    return {role: _cap(level) for role, level in plan.items()}, ", ".join(why)


def escalate(efforts: dict[str, str], *, evidence_pages: int = 0, struggling: bool = False) -> tuple[dict[str, str], str]:
    """Raise effort when the question proves harder than its classification suggested."""
    if not CONFIG.adaptive_effort:
        return efforts, ""
    new = dict(efforts)
    if struggling:                           # no evidence yet / planner tried to give up
        new["planner"] = at_least(new["planner"], "high")
    if evidence_pages >= 2:                  # evidence from several pages: may conflict or need combining
        new["answerer"] = at_least(new["answerer"], "high")
    changed = [f"{r} {efforts[r]}->{new[r]}" for r in new if new[r] != efforts[r]]
    return new, ", ".join(changed)
