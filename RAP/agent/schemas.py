"""Pydantic models, strict JSON schemas for Groq structured outputs, and the planner's
native tool definitions (one distinct function per tool)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["locate_fact", "multi_page", "comparison_or_change", "likely_unanswerable"]


class Action(BaseModel):
    action: Literal["list_headings", "search_keyword", "get_page", "finish"]
    keyword: str | None = None
    page: int | None = None
    reason: str = ""
    question_type: QuestionType | None = None
    missing: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)  # only used when USE_READER=false

    @model_validator(mode="after")
    def _args(self):
        if self.action == "search_keyword":
            if not self.keyword or not self.keyword.strip():
                raise ValueError("search_keyword requires a non-empty keyword")
            if len(self.keyword) > 100:
                raise ValueError("keyword must be at most 100 characters")
            self.keyword = self.keyword.strip()
        if self.action == "get_page" and self.page is None:
            raise ValueError("get_page requires page")
        return self


class Understanding(BaseModel):
    restated_question: str
    question_type: QuestionType
    answer_shape: str
    sub_parts: list[str]
    search_terms: list[str]
    likely_sections: list[str]
    change_risk: bool


class Quote(BaseModel):
    text: str


class ReaderOut(BaseModel):
    quotes: list[Quote] = Field(default_factory=list)
    instruction_like: bool = False
    instruction_text: str = ""      # exact injected passage, so it can never be cited as evidence
    cut_off: bool = False           # relevant text continues on the next page


class Citation(BaseModel):
    page: int
    quote: str


class FinalAnswer(BaseModel):
    status: Literal["answered", "insufficient"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Strict JSON schemas (Groq strict mode: every field required, additionalProperties false)
# --------------------------------------------------------------------------- #
_STR = {"type": "string"}
_QTYPE = {"type": "string", "enum": ["locate_fact", "multi_page", "comparison_or_change", "likely_unanswerable"]}

STRICT_SCHEMAS: dict[str, dict] = {
    "Understanding": {
        "type": "object", "additionalProperties": False,
        "required": ["restated_question", "question_type", "answer_shape", "sub_parts",
                     "search_terms", "likely_sections", "change_risk"],
        "properties": {
            "restated_question": _STR, "question_type": _QTYPE, "answer_shape": _STR,
            "sub_parts": {"type": "array", "items": _STR},
            "search_terms": {"type": "array", "items": _STR},
            "likely_sections": {"type": "array", "items": _STR},
            "change_risk": {"type": "boolean"},
        },
    },
    "ReaderOut": {
        "type": "object", "additionalProperties": False,
        "required": ["quotes", "instruction_like", "instruction_text", "cut_off"],
        "properties": {
            "quotes": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["text"],
                "properties": {"text": _STR}}},
            "instruction_like": {"type": "boolean"},
            "instruction_text": _STR,
            "cut_off": {"type": "boolean"},
        },
    },
    "FinalAnswer": {
        "type": "object", "additionalProperties": False,
        "required": ["status", "answer", "citations"],
        "properties": {
            "status": {"type": "string", "enum": ["answered", "insufficient"]},
            "answer": _STR,
            "citations": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["page", "quote"],
                "properties": {"page": {"type": "integer"}, "quote": _STR}}},
        },
    },
}


# --------------------------------------------------------------------------- #
# Planner tools: one distinct function definition per tool (OpenAI/Groq tool-calling format)
# --------------------------------------------------------------------------- #
def planner_tools(use_reader: bool = True) -> list[dict]:
    common = {
        "reason": {"type": "string", "description": "One sentence: why this is the best next step."},
        "missing": {"type": "array", "items": {"type": "string"},
                    "description": "Sub-parts of the question that still have no supporting quote."},
    }
    if not use_reader:
        common["evidence"] = {"type": "array", "items": {"type": "string"},
                              "description": "Verbatim quotes copied from pages you have read that help answer."}

    def fn(name, description, extra=None, required=None):
        props = {**(extra or {}), **common}
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props,
                           "required": (required or []) + ["reason", "missing"]}}}

    return [
        fn("list_headings",
           "Return the document's table of contents: every heading with its page number. "
           "Costs 1 tool call. Usually the best first step on documents longer than the remaining budget."),
        fn("search_keyword",
           "Return the page numbers where a keyword or exact phrase appears (case-insensitive). "
           "Returns page numbers ONLY, never text. Costs 1 tool call.",
           {"keyword": {"type": "string", "description": "One distinctive word or short phrase in the document's own wording."}},
           ["keyword"]),
        fn("get_page",
           "Read the full text of ONE page (1-based). Costs 1 tool call. The only way to obtain evidence.",
           {"page": {"type": "integer", "description": "1-based page number."}}, ["page"]),
        fn("finish",
           "Stop gathering and hand the collected evidence to the answerer. Free. Use when every sub-part "
           "has a supporting quote, when the budget is spent, or when a genuine search found nothing."),
    ]
