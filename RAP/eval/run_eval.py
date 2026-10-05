"""Run questions.jsonl through the agent; score, bucket failures, save a comparable results file.

    python eval/make_synthetic_pdf.py
    python eval/run_eval.py eval/questions.jsonl            # live (Groq)
    python eval/run_eval.py eval/questions.jsonl --ids notice caregiver
    python eval/run_eval.py eval/questions.jsonl --fake     # offline plumbing check only

questions.jsonl fields: id, pdf (relative to the jsonl), question, expected_contains (list),
expect_insufficient (bool), type, optional gold_pages (list), optional injection_marker.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import llm, prompts, tools  # noqa: E402
from agent.config import CONFIG  # noqa: E402
from agent.document import Document  # noqa: E402
from agent.loop import AgentLoop, Session  # noqa: E402
from agent.textnorm import normalize  # noqa: E402


def answer_body(text: str) -> str:
    return text.split("**Sources**")[0]


def bucket(q: dict, final: dict, events: list[dict]) -> tuple[bool, str]:
    st = final["stats"]
    body = answer_body(final["answer"])
    marker = q.get("injection_marker")
    if marker and marker in body:
        return False, "injection_followed"
    if final.get("error"):
        return False, "llm_error"
    if q["expect_insufficient"]:
        return (True, "") if final["status"] == "insufficient" else (False, "over_answer")
    norm = normalize(body)
    passed = final["status"] == "answered" and all(normalize(x) in norm for x in q["expected_contains"])
    if passed:
        return True, ""
    gold = set(q.get("gold_pages") or [])
    read = set(st["pages_read"])
    quoted_pages = {e["step_page"] for e in events if e.get("step_page") is not None}
    if final["status"] == "insufficient":
        if st["tool_calls"] >= st["budget"] and not gold <= read:
            return False, "budget_exhaustion"
        if gold and not (gold & read):
            empty_search = any(v == [] for v in st["searches"].values())
            return False, "bad_keyword" if empty_search else "wrong_tool_choice"
        if gold and not (gold & quoted_pages):
            return False, "reader_miss"
        return False, "over_abstain"
    if gold and not gold <= read:
        return False, "wrong_tool_choice"
    return False, "wrong_answer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--fake", action="store_true", help="offline heuristic LLM (plumbing check only)")
    args = ap.parse_args()

    if args.fake:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fake_llm import fake
        llm.set_fake(fake)

    qpath = Path(args.questions).resolve()
    items = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        items = [q for q in items if q["id"] in args.ids]

    docs: dict[str, Document] = {}
    rows = []
    for i, q in enumerate(items, 1):
        pdf = (qpath.parent / q["pdf"]).resolve()
        if str(pdf) not in docs:
            docs[str(pdf)] = Document.from_path(pdf)
            tools.register(docs[str(pdf)])
        doc = docs[str(pdf)]
        events = []
        for ev in AgentLoop(doc, Session()).run(q["question"]):   # fresh session: questions are independent
            if ev["type"] == "step" and ev["tool"] == "get_page" and any("quote" in f for f in ev["flags"]):
                ev["step_page"] = ev["args"]["page"]
            events.append(ev)
        final = events[-1]
        ok, why = bucket(q, final, events)
        st = final["stats"]
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"], "status": final["status"],
                     "answer": final["answer"], "pass": ok, "bucket": why, "tool_calls": st["tool_calls"],
                     "budget_violation": st["tool_calls"] > st["budget"], "pages_read": st["pages_read"],
                     "searches": st["searches"], "llm_calls": st["llm_calls"], "latency_s": st["latency_s"],
                     "expect_insufficient": q["expect_insufficient"], "question_id": final["question_id"],
                     "trace": [{k: e.get(k) for k in ("tool", "args", "status", "flags", "reason")}
                               for e in events if e["type"] == "step"]})
        print(f"[{i}/{len(items)}] {'PASS' if ok else 'FAIL'} {q['id']:<20} {st['tool_calls']} calls "
              f"{st['latency_s']}s {why}")
        if not ok:
            print("      " + answer_body(final["answer"]).replace("\n", " ")[:220])

    from report import summarize
    summary = summarize(rows)
    print(summary["text"])
    run_id = time.strftime("%Y%m%d_%H%M%S") + ("_fake" if args.fake else "")
    snapshot = {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(CONFIG).items()}
    prompt_text = "\n".join(getattr(prompts, n) for n in dir(prompts) if n.endswith("_SYSTEM"))
    out = Path(__file__).resolve().parent / f"results_{run_id}.json"
    out.write_text(json.dumps({"run_id": run_id, "llm": llm.describe(), "config": snapshot,
                               "prompts_sha": hashlib.sha256(prompt_text.encode()).hexdigest()[:12],
                               "prompts": {n: getattr(prompts, n) for n in dir(prompts) if n.endswith("_SYSTEM")},
                               "summary": {k: v for k, v in summary.items() if k != "text"}, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
