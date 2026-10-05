# Budgeted Document-Answering Agent

A chat app that answers questions about any uploaded PDF by **reading it the way a careful person
would**: look at the table of contents, search for a word, open a few pages. It has a hard limit of
**6 tool calls per question** and says **"Insufficient information"** instead of guessing.

No RAG, no embeddings, no vector database, no agent framework.

**Contents:** [Overview](#1-overview) · [Quick start](#2-quick-start) · [Architecture](#3-architecture) ·
[Flow](#4-flow-how-a-question-is-answered) · [Design decisions](#5-design-decisions) ·
[Configuration](#6-configuration) · [File structure](#7-file-structure) · [Testing](#8-testing-and-evaluation) ·
[Logging](#9-logging-and-the-trace) · [Limitations](#10-known-limitations)

---

## 1. Overview

### The problem
Answer questions from a PDF the agent has never seen, using **only** four tools:

| Tool | Returns |
|---|---|
| `list_documents()` | Titles and metadata only |
| `list_headings(doc_id)` | The table of contents: headings with page numbers |
| `get_page(doc_id, page)` | The text of exactly one page |
| `search_keyword(doc_id, keyword)` | Page numbers where the keyword appears, nothing else |

**Rules:** at most 6 tool calls per question (plus one final answer), no embeddings or semantic search,
no direct document access, no agent frameworks, no caching pages across questions, and decline rather
than guess. The live test PDF contains answers spread over pages, statements later superseded,
questions with no answer, and text that tries to redirect the agent.

### What this project does
- Upload any text-based PDF and ask questions in a chat.
- Each question is first **analysed** (what is asked, what a full answer needs, which words to search).
- A **planner** spends up to 6 tool calls finding the right pages; a separate **reader** copies exact
  quotes; **code** checks every quote; an **answerer** writes the reply from verified quotes only.
- Every answer shows its **sources** (page + exact quote). Every tool call and LLM call is **logged**
  and shown live in a trace panel.

### Key properties
| Property | How it is achieved |
|---|---|
| Never exceeds 6 calls | A code gate refuses the 7th call; proven by a test |
| Does not guess | An answer survives only if its citations are found verbatim in pages actually read |
| Resists injected instructions | The model that reads the page has no tools; the model with tools never reads pages |
| Handles amended values | Questions about values that change trigger a check of later pages / amendments |
| Transparent | Live trace in the UI; full JSONL log of every call |

### Tech stack
Python 3.11+ · PyMuPDF (PDF text) · Groq API via the `openai` SDK (`openai/gpt-oss-120b`,
`openai/gpt-oss-20b`) · Pydantic (validation) · Gradio (UI) · pytest.

---

## 2. Quick start

```bash
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
# open .env and set GROQ_API_KEY=gsk_...   (never commit .env or share the key)
python app.py                   # open http://127.0.0.1:7860
```

The app reads `.env` itself; no shell `export` / `$env:` commands are needed.

**Using the app:** upload a PDF (top left) → wait for the line with title and page count → type a
question and press Enter. The right panel shows how the question was understood, the reasoning effort
chosen, and each tool call as it happens. **Download trace** exports the session log;
**Reset session** clears the chat but keeps the PDF.

---

## 3. Architecture

<img width="1568" height="640" alt="architecture" src="https://github.com/user-attachments/assets/84594a4c-c240-4298-a13c-87454b93307e" />

Blue = LLM role, grey = plain code. The orange arrow is the **only** path raw page text travels:
into the Reader, which has no tools.

### Components

| # | Component | Type | Responsibility | File |
|---|---|---|---|---|
| 1 | **Understand** | LLM (strong model) | Before any tool call: restated question, type, answer shape, sub-parts, search terms in the document's wording, whether the value may be amended | `agent/planner.py` |
| 2 | **Effort policy** | Code | Picks reasoning effort per role from the question type; raises it mid-question if needed | `agent/effort.py` |
| 3 | **Planner** | LLM (strong model) | Chooses one native tool call per step; tracks what is still missing; decides when to finish | `agent/planner.py` |
| 4 | **ToolGate** | Code | Validates, de-duplicates, counts and logs every call; refuses the 7th | `agent/gate.py` |
| 5 | **Tools** | Code | The four permitted tools over the in-memory PDF | `agent/tools.py` |
| 6 | **Reader** | LLM (fast model), no tools | Reads one page; returns up to 3 verbatim quotes; flags injected instructions and cut-off text | `agent/reader.py` |
| 7 | **Quote check** | Code | Keeps quotes found on the page; repairs near-misses; drops injected text | `agent/verify.py` |
| 8 | **Answerer** | LLM (strong model), once | Writes the answer from verified quotes only; resolves conflicts in favour of amendments | `agent/answerer.py` |
| 9 | **Citation check** | Code | Re-verifies every citation; none valid → "Insufficient information" | `agent/verify.py` |
| 10 | **Call log** | Code | Records every tool and LLM call; adapter for the organizers' wrapper | `agent/calllog.py` |

### Who can read, who can act

<img width="1568" height="908" alt="read-vs-act" src="https://github.com/user-attachments/assets/2d5b8eae-ef46-48e5-8573-2ee8fd149bb6" />

| Component | Sees the question | Sees raw page text | Can call tools |
|---|---|---|---|
| Understand | Yes | No (title and page count only) | No |
| Planner | Yes, plus the analysis | **No** | **Yes** |
| Reader | Yes | **Yes**, one page at a time | **No** |
| Answerer | Yes | No (verified quotes only) | No |
| Gate and checks (code) | n/a | Yes, to verify quotes | Enforce the budget |

---

## 4. Flow: how a question is answered

<img width="2064" height="1450" alt="question-flow" src="https://github.com/user-attachments/assets/2987c792-2548-4d18-a20f-92fad32541b2" />

*Expected path for one question on the synthetic test PDF: the rule (30 days) is on page 7 and its
amendment (45 days) on page 22. 4 of 6 tool calls used.*

### Step by step

1. **Upload.** The PDF is opened in memory from its bytes. **No text is extracted.** Text only ever
   comes from a counted `get_page` or `search_keyword` call.
2. **New question.** A fresh question id, a fresh gate with 6 calls, and empty per-question state.
3. **Understand** (0 tool calls). The LLM sees only the question, earlier questions in the chat, and the
   document title and page count, and returns strict JSON:
   ```json
   {"restated_question": "What notice period must an employee give when resigning?",
    "question_type": "comparison_or_change", "answer_shape": "a number of days",
    "sub_parts": ["current notice period for resignation"],
    "search_terms": ["notice period", "resignation", "addendum"],
    "likely_sections": ["Resignation", "Addendum"], "change_risk": true}
   ```
4. **Effort policy.** `comparison_or_change` → planner high, reader low, answerer high.
5. **Plan → act loop** (repeats until `finish` or 6 calls):
   - The **planner** receives a freshly built context every step (question analysis, budget left,
     calls so far, headings, search hits, verified quotes, unread relevant pages, what is missing) and
     returns one native tool call, e.g. `get_page(page=7, reason=..., missing=[...])`.
   - The **ToolGate** validates and executes it. Duplicates and invalid pages are rejected for free and
     reported back to the planner.
   - After `get_page`, the **reader** returns exact quotes; the **quote check** keeps, repairs or drops them.
   - If the planner tries to finish with **no evidence** while relevant pages are unread, it is nudged
     once and its effort is raised.
6. **Answer.** If evidence came from 2+ pages, answerer effort is raised. The answerer sees only verified
   quotes and returns `{status, answer, citations}`.
7. **Citation check.** Each citation must be found in a page read in this question and not in flagged
   injected text. None valid → "Insufficient information" plus what was searched.
8. **Output.** Answer with sources, a stats line (tool calls, LLM calls, time), the trace table, and log
   entries. Per-question state is discarded; only the question text is kept for follow-ups.

### How the four demo cases are handled

| Case | Handling |
|---|---|
| Answer spans pages | Sub-parts tracked in `missing` until each has a quote; the reader's `cut_off` flag sends the planner to the next page |
| Later text supersedes earlier | Amendable values trigger a check of the Addendum or last search hit; the answer gives the current value and the one it replaced, citing both |
| No answer in the document | No verified quote → "Insufficient information", stating what was searched |
| Text that tries to redirect the agent | Only the tool-less reader sees it; it is flagged, its exact span recorded, and it can never be cited |

---

## 5. Design decisions

| Decision | Why | Where |
|---|---|---|
| **Budget enforced in code**, not by prompt | A 7th call fails the question; code makes it impossible rather than unlikely | `gate.py` |
| **Understand the question before any call** | Defines "done" (sub-parts) up front and picks search words in the document's wording, since search is exact-match | `planner.py` |
| **Headings carry page numbers** | A table of contents is the cheapest way to target a page; PDF outline first, font-size/bold heuristic as fallback | `tools.py` |
| **Native tool calling** for the planner (one function per tool) | Removes hand-written-JSON failures; each call has typed arguments | `schemas.py`, `llm.py` |
| **Strict JSON-schema output** for understand, reader, answerer | Guaranteed parseable; falls back to JSON mode, then prompt-only JSON | `llm.py` |
| **Quarantine: planner never reads pages; reader has no tools** | Injected instructions can only reach a model that cannot act | `loop.py`, `reader.py` |
| **Random per-session delimiters + stripping of tag/role markers** | Page text cannot close its wrapper or impersonate system/user turns | `sanitize.py` |
| **Evidence verified in code, not by another LLM** | Deterministic; guessing is penalised more than declining | `verify.py` |
| **Near-miss quotes repaired, not dropped** (all numbers must match) | Small copy differences used to throw away good evidence and cause empty answers | `verify.py` |
| **Injected text can never be cited** | A quote can be verbatim and still be the attacker's words | `reader.py`, `verify.py` |
| **Nudge when finishing with no evidence** | Stops premature "insufficient" while relevant pages are unread | `loop.py` |
| **Adaptive reasoning effort** with a ceiling | Simple questions stay fast; hard ones (amended values, multi-part) think harder | `effort.py` |
| **Fresh state per question**; only question text kept | Honours "no caching to bypass the budget" while allowing follow-up questions | `loop.py` |
| **Every call logged, errors shown** | Nothing hidden; failures are visible instead of silent "insufficient" | `calllog.py`, `app.py` |
| **Assumption:** only document tools count toward the budget | LLM calls are not tool calls; confirm with organizers | — |

---

## 6. Configuration

All settings live in `.env` (copy from `.env.example`). Restart `python app.py` after changes.
Real environment variables override `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `gemini`, `anthropic` |
| `GROQ_API_KEY` | | API key (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` for others) |
| `MODEL_STRONG` | `openai/gpt-oss-120b` | Understand, planner, answerer |
| `MODEL_FAST` | `openai/gpt-oss-20b` | Reader |
| `MAX_TOOL_CALLS` | `6` | Hard budget per question (competition rule: keep 6) |
| `MAX_PLANNER_STEPS` | `10` | Loop guard; rejected calls and nudges use a step, not budget |
| `MAX_PAGE_CHARS` | `12000` | Longer pages are truncated and marked `[TRUNCATED]` |
| `MAX_HEADINGS` | `200` | Longer tables of contents are truncated and marked |
| `USE_READER` | `true` | `false` lets the planner read pages itself (only if LLM calls must be minimised) |
| `TEMPERATURE` | `0` | Same behaviour in the demo as in testing |
| `ADAPTIVE_EFFORT` | `true` | Choose reasoning effort per question |
| `REASONING_EFFORT_CEILING` | `high` | Cap on any effort; set `medium` if the demo is slow |
| `REASONING_EFFORT_UNDERSTAND` | `low` | Effort for the understand step |
| `REASONING_EFFORT_PLANNER` / `_READER` / `_ANSWERER` | `medium` / `low` / `medium` | Used only when `ADAPTIVE_EFFORT=false` |
| `LLM_TIMEOUT_S` | `60` | Seconds to wait for one LLM reply |
| `CALL_LOG_PATH` | `logs/calls.jsonl` | Where the trace is written |
| `HOST` / `PORT` | `127.0.0.1` / `7860` | Where the UI is served |

**Adaptive effort profiles** (`agent/effort.py`):

| Question type | Planner | Reader | Answerer |
|---|---|---|---|
| `locate_fact` | low | low | medium |
| `multi_page` | medium | low | high |
| `comparison_or_change` | high | low | high |
| `likely_unanswerable` | medium | low | medium |

Raised automatically: value may be amended → planner ≥ medium; 3+ sub-parts → planner ≥ medium,
answerer high; no evidence when finishing → planner high; evidence from 2+ pages → answerer high.
Everything is capped by `REASONING_EFFORT_CEILING`.

Fixed in code (`agent/config.py`): at most 3 quotes per page, 400 characters per quote.

---

## 7. File structure

```
doc-agent-v2/
├── app.py                    Gradio UI: upload, chat, live trace, reset, download trace
├── requirements.txt          pymupdf, openai, pydantic, gradio, pytest
├── .env.example              Settings template (copy to .env; never commit .env)
├── README.md
├── agent/
│   ├── config.py             Settings from .env / environment
│   ├── document.py           PDF opened from bytes; 1-based pages; no text at upload
│   ├── tools.py              list_documents, list_headings, get_page, search_keyword
│   ├── textnorm.py           Text normalization shared by search and quote checks
│   ├── gate.py               ToolGate: budget, duplicates, validation, logging
│   ├── calllog.py            JSONL logger + adapter for the organizers' wrapper
│   ├── llm.py                Provider client: native tool calls, strict JSON, retries, back-off
│   ├── schemas.py            Pydantic models, strict JSON schemas, planner tool definitions
│   ├── prompts.py            System prompt for each LLM role
│   ├── sanitize.py           Per-session delimiters; strips tag/role-marker mimics
│   ├── planner.py            Understand step and planner
│   ├── reader.py             Quote extraction from one page (no tools)
│   ├── verify.py             Quote check, quote repair, final citation check
│   ├── answerer.py           The single final answer call
│   ├── effort.py             Adaptive reasoning effort policy
│   └── loop.py               AgentLoop: the per-question workflow; streams trace events
├── docs/                     Diagrams used in this README (PNG + SVG source)
├── tests/                    48 offline tests (no network; scripted LLM)
│   ├── conftest.py
│   ├── test_m1_tools_gate.py
│   ├── test_m2_llm_verify.py
│   ├── test_m3_loop.py
│   └── test_effort.py
├── eval/                     Optional evaluation harness
│   ├── make_synthetic_pdf.py 24-page test PDFs containing every demo trap
│   ├── questions.jsonl       16 test questions with expected answers
│   ├── run_eval.py           Runs the agent on questions; scores and labels failures
│   ├── report.py             Summarises and compares runs
│   └── fake_llm.py           Offline stand-in for plumbing checks only (optional)
└── logs/
    └── calls.jsonl           Full trace of every tool and LLM call (created at runtime)
```

---

## 8. Testing and evaluation

```bash
python -m pytest -q                                          # 48 tests, no network
python eval/make_synthetic_pdf.py                            # builds eval/data/*.pdf
python eval/run_eval.py eval/questions.jsonl                 # live run on Groq
python eval/run_eval.py eval/questions.jsonl --ids notice caregiver
python eval/report.py eval/results_A.json eval/results_B.json   # compare two runs
```

- **Unit and loop tests** cover normalization, 1-based paging, search, headings (outline and fallback),
  the gate (6 allowed, 7th refused, duplicates free), sanitizing, quote and citation checks,
  structured-output retries, rate-limit back-off, tool-call parsing, adaptive effort, and the full
  loop with a scripted LLM (budget, nudges, injection, no state carried between questions).
- **Evaluation** runs 16 questions on a 24-page synthetic handbook (with and without a PDF outline)
  covering a page-split answer, two amended values, an injected instruction, a fact under an
  unrelated heading, and absent facts. Each failure is labelled (`wrong_tool_choice`, `bad_keyword`,
  `reader_miss`, `over_abstain`, `over_answer`, `budget_exhaustion`, `injection_followed`,
  `wrong_answer`, `llm_error`); each run is saved with its settings and prompts.
- Add the organizers' sample PDFs to `eval/data/` and their questions to `questions.jsonl`.

---

## 9. Logging and the trace

Every tool attempt (executed or rejected) and every LLM call (role, model, reasoning effort, output
preview, tokens, duration) is written to `logs/calls.jsonl` with session and question ids. The UI's
trace table matches the log one-to-one for executed tools.

**Organizers' wrapper:** edit the two `TODO(wire-organizer-wrapper)` markers in `agent/calllog.py`:
apply it in `organizer_wrap()` if it is a decorator for tool functions (all four tools pass through
it), or forward events in `OrganizerWrapperLogger.log()` if it is an event logger.

**For the live demo:** delete `logs/` beforehand, run the demo, then submit `logs/calls.jsonl`
(or use **Download trace**) as the full trace.

---

## 10. Known limitations

| Limitation | Effect | Intended fix |
|---|---|---|
| Images and scanned pages | Only the text layer is read; images, charts and scanned pages are invisible (`[NO_TEXT_LAYER]`) | OCR inside `get_page`, plus a vision model for charts |
| Paraphrases | Exact search misses synonyms; each extra search costs a call | More word-form folding in `search_keyword`; lean on headings |
| Evidence on 5+ pages | Cannot fit 6 calls | Sharper page targeting; today it answers the supported part and names what is missing |
| Page-break splits | Missed if the next page is not read | Controller reads page n+1 automatically on a cut-off flag when budget remains |
| Injection | Mitigated, not guaranteed: an unflagged injected sentence is still verbatim | Second detector (rules or classifier) on quotes |
| Heading heuristic | Can misfire on PDFs with no outline and uniform fonts | Switch to search sooner |
| Printed vs PDF page numbers | "See page 12" may not be PDF page 12 | Detect the offset from footers |
| Tables / columns | Read as plain text; reading order can scramble | Layout-aware extraction |
| Latency / rate limits | About 8–14 LLM calls per question | Lower effort ceiling; provider fallback via `LLM_PROVIDER` |

---

### Before submitting
- [ ] Organizers' wrapper wired in `agent/calllog.py`
- [ ] `.env` **not** included (submit `.env.example`)
- [ ] `logs/` cleared before the demo; demo `logs/calls.jsonl` included as the trace
- [ ] Memo filled in with eval numbers
- [ ] `__pycache__/`, `.pytest_cache/`, practice results removed
