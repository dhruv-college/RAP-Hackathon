# Budgeted Document-Answering Agent (v2)

Upload a PDF, ask questions. The agent reads the document only through four tools, with a
hard limit of **6 tool calls per question** plus one final answer call, and says
"Insufficient information" rather than guess. No RAG, no embeddings, no agent framework.

## Setup (Windows PowerShell or bash)

```
pip install -r requirements.txt
copy .env.example .env        # bash: cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
python app.py                 # open http://127.0.0.1:7860
```

The app reads `.env` itself, so there is no need to set environment variables in the shell.

## Test and evaluate

```
python -m pytest -q                                   # unit + loop tests, no network
python eval/make_synthetic_pdf.py                     # 24-page PDFs with every judged trap
python eval/run_eval.py eval/questions.jsonl          # live run on Groq
python eval/run_eval.py eval/questions.jsonl --ids notice caregiver
python eval/report.py eval/results_A.json eval/results_B.json   # compare two runs
```

Each run saves `eval/results_<runid>.json` with the config and prompt snapshot, so tuning runs
are comparable. Failure buckets: `wrong_tool_choice`, `bad_keyword`, `reader_miss`, `over_abstain`,
`over_answer`, `budget_exhaustion`, `injection_followed`, `wrong_answer`, `llm_error`.

## Architecture

```
Gradio UI -- upload --> Document (opened from bytes; no text extracted at upload)
   |
   +-- question --> AgentLoop (one per question)
        1. Understand (LLM, 0 tool calls): restated question, answer shape, sub-parts,
           search terms in the document's wording, change risk
        2. Planner (LLM, native tool calling: one distinct function per tool) -> one Action
           ToolGate (code): validates, rejects duplicates, counts, logs; the 7th call is impossible
           Tools: list_headings | search_keyword | get_page (| list_documents)
           Reader (LLM, NO tools): verbatim quotes from the page just read
           Verifier (code): quote must be in the page; near-misses repaired to the page sentence
        3. Answerer (LLM, exactly once): answer from verified quotes only
        4. check_final (code): every citation verified; none valid -> "insufficient"
```

- **Quarantine.** The planner never sees raw page text; the reader sees it but has no tools; the
  answerer sees only verified quotes.
- **Structural budget.** `ToolGate` is the single point of enforcement. Duplicate or invalid calls
  are rejected without spending budget. When the budget is spent, the loop goes straight to the answerer.
- **Native tool calling.** Each tool is its own function definition (`agent/schemas.py:planner_tools`),
  so the planner emits a real `get_page(page=4)` call. Reader, understanding and answerer use Groq's
  strict JSON-schema mode, falling back to JSON mode, then prompt-only JSON if a provider rejects it.
- **Understand first.** A question-analysis step runs before any tool call and is shown to the
  planner on every step, so "done" is defined by the sub-parts, not by a feeling.
- **Adaptive reasoning effort.** The understand step classifies each question; `agent/effort.py` maps the
  class to an effort per role (simple fact: planner low, answerer medium; changed value: planner and
  answerer high) and raises it mid-question if the planner finds no evidence or the evidence spans several
  pages. `REASONING_EFFORT_CEILING` caps it; `ADAPTIVE_EFFORT=false` returns to fixed values. The effort
  used for every LLM call is logged and shown in the UI.
- **Evidence-required answers.** Every claim needs a verbatim quote with its page, checked in code.
- **No silent failures.** LLM errors appear in the chat and in the trace; they are not turned into
  a quiet "insufficient".

### What changed from v1 (why v1 often gave no answer)

| v1 behaviour | v2 |
|---|---|
| Planner wrote JSON by hand; malformed output fell back silently | Native tool calls; one corrective retry, then a visible finish |
| Extractor errors were swallowed and became "no evidence" | Reader errors are reported in the trace |
| Quotes not matching exactly were dropped | Near-miss quotes are repaired to the verbatim page sentence (all numbers must match) |
| A separate LLM verifier could veto supported answers | Verification is pure code (spec) |
| Planner could finish with zero evidence while relevant pages were unread | One nudge listing the unread candidate pages |
| `max_tokens` capped output, which can truncate a reasoning model's JSON | No output cap; provider default |

## Logging

Every tool attempt (executed or rejected) and every LLM call goes to `logs/calls.jsonl`
(`agent/calllog.py`), including the LLM's raw output preview, so a bad answer can be diagnosed.
"Download trace" in the UI exports the current session. To wire the organizers' wrapper, see the
two `TODO(wire-organizer-wrapper)` markers in `agent/calllog.py`.

## Known limitations

- No OCR: scanned pages return `[NO_TEXT_LAYER]` and the agent says it cannot read them.
- Exact-match keyword search misses paraphrases; each retry costs one call.
- Answers needing more than about 3 to 4 pages of evidence may hit the budget.
- Facts split across a page boundary can be missed if only one page is read (the reader's
  `cut_off` flag prompts reading the next page, but only if budget remains).
- The heading fallback heuristic can misfire on unusual layouts.
- Injection defense is layered mitigation, not a guarantee: role separation, random per-session
  delimiters, delimiter and role-marker stripping, data-only instructions, and quote verification
  that refuses quotes from passages the reader flagged. An injected sentence the reader fails to
  flag is still a verbatim quote.
- Single-provider rate limits (switch with `LLM_PROVIDER`).
- Quote repair accepts a page sentence containing all the quote's numbers and at least 80% of its
  words; a wrong sentence with the same numbers could in principle be chosen.
