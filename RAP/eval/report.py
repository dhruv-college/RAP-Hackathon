"""Summaries for eval results.

    python eval/report.py eval/results_A.json                  # one run
    python eval/report.py eval/results_A.json eval/results_B.json   # compare two runs
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if not r["expect_insufficient"]]
    una = [r for r in rows if r["expect_insufficient"]]
    out = {
        "questions": len(rows),
        "accuracy_answerable": f"{sum(r['pass'] for r in ans)}/{len(ans)}",
        "correct_abstention": f"{sum(r['pass'] for r in una)}/{len(una)}",
        "wrong_answer_on_unanswerable": sum(1 for r in una if r["status"] == "answered"),
        "budget_violations": sum(r["budget_violation"] for r in rows),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / max(1, len(rows)), 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / max(1, len(rows)), 1),
        "failure_buckets": dict(Counter(r["bucket"] for r in rows if not r["pass"])),
        "by_type": {t: f"{sum(r['pass'] for r in rows if r['type'] == t)}/{sum(1 for r in rows if r['type'] == t)}"
                    for t in sorted({r["type"] for r in rows})},
    }
    lines = ["", "=== Summary ==="] + [f"{k:<30} {v}" for k, v in out.items()]
    out["text"] = "\n".join(lines)
    return out


def main(paths: list[str]) -> None:
    runs = [json.load(open(p, encoding="utf-8")) for p in paths]
    for run in runs:
        print(f"\n# {run['run_id']}  ({run['llm']}, prompts {run['prompts_sha']})")
        print(summarize(run["rows"])["text"])
    if len(runs) == 2:
        a, b = ({r["id"]: r for r in run["rows"]} for run in runs)
        print("\n=== Changed outcomes ===")
        for qid in sorted(set(a) & set(b)):
            if a[qid]["pass"] != b[qid]["pass"]:
                print(f"{qid:<22} {'PASS' if a[qid]['pass'] else 'FAIL'} -> {'PASS' if b[qid]['pass'] else 'FAIL'} "
                      f"({b[qid]['bucket'] or 'ok'})")


if __name__ == "__main__":
    main(sys.argv[1:])
