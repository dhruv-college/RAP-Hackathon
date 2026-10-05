# Budgeted Document-Answering Agent

Upload a PDF, ask questions in a chat. The agent reads the document **only** through four tools,
with a hard limit of **6 tool calls per question** plus one final answer call, and answers
**"Insufficient information"** rather than guess.

No RAG, no embeddings, no vector store, no agent framework: plain Python, PyMuPDF, Groq, Gradio.

---

## Quick start

```bash
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
# open .env and set GROQ_API_KEY=gsk_...   (never commit .env or paste the key anywhere)
python app.py                   # open http://127.0.0.1:7860
```

The app reads `.env` itself, so no shell `export` / `$env:` commands are needed.

**Using it:** upload a PDF (top left), wait for the line showing title and page count, then type a
question and press Enter. The right-hand panel shows how the question was understood, the reasoning
effort chosen, and every tool call as it happens (`used/6`, result, the planner's reason, flags).
**Download trace** exports the session's full log; **Reset session** clears the chat but keeps the PDF.

---

## Architecture

![Architecture: understand, effort policy, plan-act loop with ToolGate, tools, Reader and quote check, then answerer and citation check](docs/architecture.png)

Blue boxes are LLM roles, grey boxes are plain code. The orange arrow is the **only** path raw page
text travels: into the Reader, which has no tools.

## How a question is answered

The sequence below is the expected path for one question on the synthetic test PDF: the original rule
(30 days) is on page 7 and an amendment (45 days) on page 22. It uses 4 of the 6 tool calls.

![Question flow: list_headings, search_keyword, get_page 7 and 22, reader quotes verified, answerer returns 45 days replacing 30](docs/question-flow.png)

1. **Upload.** The PDF is opened in memory from its bytes. **No text is extracted.** Text is only
   ever produced by a counted `get_page` or `search_keyword` call.
2. **Understand** (`agent/planner.py:understand`). Before any tool call, an LLM reads only the
   question (plus earlier questions in the chat, title and page count) and returns: the restated
   question, question type, what a complete answer contains, its sub-parts, search terms in the
   document's likely wording, likely section headings, and whether the value is the kind documents
   later amend.
3. **Effort policy** (`agent/effort.py`). The question type sets reasoning effort per role, for
   example simple fact: planner low; value that may have changed: planner and answerer high.
4. **Plan → act loop** (`agent/loop.py`), repeated until the planner calls `finish` or 6 calls are spent:
   - **Planner** (LLM) chooses one **native tool call**: `list_headings`, `search_keyword(keyword)`,
     `get_page(page)` or `finish`, with a one-line reason and a list of what is still missing.
     It sees the question analysis, headings, search hits and verified quotes, **never raw page text**.
   - **ToolGate** (code) validates the call, rejects duplicates and out-of-range pages for free,
     counts it, logs it, and **refuses a 7th call**.
   - After `get_page`, the **Reader** (LLM, **no tools**) reads that one page and copies up to 3
     verbatim quotes, flagging instruction-like text and content cut off at the page end.
   - **Quote check** (code) keeps a quote only if it is on the page; a near-miss is repaired to the
     exact page sentence if all its numbers match; quotes from flagged injected text are dropped.
   - If the planner tries to finish with **no evidence** while relevant pages are unread, it is
     nudged once and its effort is raised.
5. **Answerer** (LLM, exactly once) writes the answer from verified quotes only. It prefers an explicit
   amendment or the later page and mentions the value it replaced.
6. **Citation check** (code). Every citation must come from a page read in this question and not
   from injected text. No valid citation means "Insufficient information", followed by what was searched.
7. Per-question state is discarded. Only the question text is kept, so follow-ups make sense.

### Who sees what

![The model that acts never reads the document; the model that reads cannot act](docs/read-vs-act.png)

| Component | Sees the question | Sees raw page text | Can call tools |
|---|---|---|---|
| Understand | Yes | No (title and page count only) | No |
| Planner | Yes, plus the analysis | **No** | **Yes** |
| Reader | Yes | **Yes**, one page at a time | **No** |
| Answerer | Yes | No (verified quotes only) | No |
| Gate and checks (code) | n/a | Yes, to verify quotes | Enforce the budget |

The model that chooses actions never reads the document, and the model that reads the document cannot act.

### How the four demo cases are handled

| Case | Handling |
|---|---|
| Answer spans pages | Sub-parts tracked in `missing` until each has a quote; the Reader's `cut_off` flag sends the planner to the next page |
| Later text supersedes earlier | Amendable values trigger a check of the Addendum or the last search hit; the answer gives the current value and the one it replaced, citing both |
| No answer in the document | No verified quote → "Insufficient information", stating what was searched |
| Text that tries to redirect the agent | Only the tool-less Reader sees it; it is flagged, its exact span recorded, and it can never be cited |

---

## Design decisions

- **Budget enforced in code.** `ToolGate` is the single enforcement point; a test proves a 7th call is impossible.
- **Understand first.** "Done" is defined by sub-parts before any budget is spent, and search terms are
  chosen in document wording because search is exact-match.
- **Headings with page numbers** (PDF outline, else a font-size/bold heuristic) often reach the right
  page without a search.
- **Native tool calling** for the planner (one function per tool); **strict JSON-schema output** for the
  understand step, reader and answerer, falling back to JSON mode, then prompt-only JSON.
- **Quarantine** against prompt injection, plus random per-session delimiters and stripping of tag and
  role-marker sequences in all document text.
- **Evidence-required answers**, verified in code, not by another LLM.
- **Adaptive reasoning effort** per question, with a configurable ceiling for latency.
- **No silent failures.** LLM errors appear in the chat and trace instead of a quiet "insufficient".
- **Assumption:** only the four document tools count toward the budget; LLM calls do not.

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `gemini`, `anthropic` |
| `GROQ_API_KEY` | | Key for the chosen provider (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) |
| `MODEL_STRONG` | `openai/gpt-oss-120b` | Understand, planner, answerer |
| `MODEL_FAST` | `openai/gpt-oss-20b` | Reader |
| `MAX_TOOL_CALLS` | `6` | Hard budget per question (competition rule; keep at 6) |
| `MAX_PLANNER_STEPS` | `10` | Loop guard; rejected calls and nudges use a step, not budget |
| `MAX_PAGE_CHARS` | `12000` | Longer pages are truncated and marked `[TRUNCATED]` |
| `MAX_HEADINGS` | `200` | Longer tables of contents are truncated and marked |
| `USE_READER` | `true` | `false` lets the planner read pages itself (only if LLM calls must be minimised) |
| `TEMPERATURE` | `0` | Same behaviour in the demo as in testing |
| `ADAPTIVE_EFFORT` | `true` | Choose reasoning effort per question (`agent/effort.py`) |
| `REASONING_EFFORT_CEILING` | `high` | Cap on any effort; set `medium` if the demo is slow |
| `REASONING_EFFORT_UNDERSTAND` | `low` | Effort for the understand step (always used) |
| `REASONING_EFFORT_PLANNER` / `_READER` / `_ANSWERER` | `medium` / `low` / `medium` | Used only when `ADAPTIVE_EFFORT=false` |
| `LLM_TIMEOUT_S` | `60` | Seconds to wait for one LLM reply |
| `CALL_LOG_PATH` | `logs/calls.jsonl` | Where the trace is written |
| `HOST` / `PORT` | `127.0.0.1` / `7860` | Where the UI is served |

Restart `python app.py` after changing `.env`.

---

## Project structure

```
app.py                  Gradio UI: upload, chat, live trace, reset, download trace
agent/
  config.py             Settings from .env / environment
  document.py           PDF opened from bytes; 1-based pages; no text at upload
  tools.py              list_documents, list_headings, get_page, search_keyword
  textnorm.py           Normalization shared by search and quote checks
  gate.py               ToolGate: budget, duplicates, validation, logging
  calllog.py            JSONL logger + adapter for the organizers' wrapper
  llm.py                Provider client, native tool calls, strict JSON, retries
  schemas.py            Pydantic models, strict JSON schemas, planner tool definitions
  prompts.py            System prompts for each LLM role
  sanitize.py           Per-session delimiters, stripping of tag/role-marker mimics
  planner.py            Understand step and planner
  reader.py             Quote extraction from one page (no tools)
  verify.py             Quote checks, quote repair, final citation check
  answerer.py           The single final answer call
  effort.py             Adaptive reasoning effort policy
  loop.py               AgentLoop: the per-question workflow, streams trace events
docs/                   README diagrams (PNG for display, SVG sources)
tests/                  48 offline tests (no network, scripted LLM)
eval/                   Optional test harness (see below)
logs/calls.jsonl        Full trace of every tool and LLM call
```

---

## Testing and evaluation

```bash
python -m pytest -q                                          # 48 tests, no network
python eval/make_synthetic_pdf.py                            # 24-page PDFs with every demo trap
python eval/run_eval.py eval/questions.jsonl                 # live run on Groq
python eval/run_eval.py eval/questions.jsonl --ids notice caregiver
python eval/report.py eval/results_A.json eval/results_B.json   # compare two runs
```

- **Tests** cover normalization, 1-based paging, search, headings (outline and fallback), the gate
  (6 allowed, 7th refused, duplicates free), sanitizing, quote and citation checks, structured-output
  retries, rate-limit back-off, native tool-call parsing, adaptive effort, and the full loop with a
  scripted LLM (budget, nudges, injection, no state carried between questions).
- **Eval** runs `questions.jsonl` through the real agent: 16 questions on a 24-page synthetic handbook,
  with and without a PDF outline, covering a page-split answer, two amended values, an injected
  instruction, a fact under an unrelated heading, and absent facts. Each failure is labelled
  (`wrong_tool_choice`, `bad_keyword`, `reader_miss`, `over_abstain`, `over_answer`,
  `budget_exhaustion`, `injection_followed`, `wrong_answer`, `llm_error`), and each run is saved
  with its settings and prompts.
- Add the organizers' sample PDFs to `eval/data/` and their questions to `questions.jsonl`
  (`id, pdf, question, expected_contains, expect_insufficient, type, gold_pages`).
- `run_eval.py --fake` uses `eval/fake_llm.py`, a crude offline stand-in, only to check the eval
  plumbing. Its scores are meaningless; delete the file if not needed.

---

## Logging and the organizers' wrapper

Every tool attempt (executed or rejected) and every LLM call (role, model, reasoning effort, output
preview, tokens, duration) is written to `logs/calls.jsonl` with session and question ids. The UI's
trace table and the log agree one-to-one with the tools actually executed.

To connect the **provided call-logging wrapper**, edit the two `TODO(wire-organizer-wrapper)`
markers in `agent/calllog.py`:
- if it is a **decorator** for tool functions, apply it in `organizer_wrap()`; all four tools pass through it;
- if it is an **event logger**, forward events in `OrganizerWrapperLogger.log()`.

---

## Before submitting

- [ ] Organizers' wrapper wired in `agent/calllog.py`
- [ ] `.env` **not** included (submit `.env.example`)
- [ ] `logs/` cleared before the live demo, then `logs/calls.jsonl` from the demo included as the trace
- [ ] Memo filled in with eval numbers
- [ ] `__pycache__/`, `.pytest_cache/` and practice results removed

---

## Known limitations

- **Scanned PDFs:** no OCR, so pages return `[NO_TEXT_LAYER]` and the agent says it cannot read them.
- **Paraphrases:** exact keyword search misses synonyms; each extra search costs a call.
- **Large answers:** evidence spread over more than about 4 to 5 pages may not fit in 6 calls; the
  agent answers the supported part and states what is missing.
- **Page breaks:** a fact split across pages is missed if the planner does not read the next page.
- **Headings:** the fallback heuristic can misfire on PDFs with no outline and uniform fonts.
- **Injection:** layered mitigation, not a guarantee. An injected sentence the Reader fails to flag
  is still a verbatim quote.
- **Page numbers:** printed page numbers ("see page 12") may differ from PDF page indices.
- **Quote repair** accepts a page sentence with all the quote's numbers and at least 80% of its words;
  a different sentence with the same numbers could in principle be chosen.
- **Latency and rate limits:** about 8 to 14 LLM calls per question; lower `REASONING_EFFORT_CEILING`
  or switch `LLM_PROVIDER` if needed.
