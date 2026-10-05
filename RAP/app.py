"""Gradio UI: upload a PDF, ask questions, watch every tool call stream into the trace panel.

Run:  python app.py    then open http://127.0.0.1:7860
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import gradio as gr

from agent import calllog, llm, tools
from agent.config import CONFIG, ROOT
from agent.document import Document
from agent.loop import AgentLoop, Session

TRACE_HEADERS = ["#", "tool", "args", "result", "budget", "status / flags", "planner's reason"]


def _chatbot(**kw):
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:  # Gradio 5
        kw["type"] = "messages"
    return gr.Chatbot(**kw)


def _new_state() -> dict:
    return {"session": Session(), "doc_id": None}


def _doc_info(doc: Document | None) -> str:
    key = "" if CONFIG.api_key or llm._fake else f" · **{CONFIG.llm_provider.upper()} API key missing: add it to .env**"
    base = f"LLM: {llm.describe()} · budget {CONFIG.max_tool_calls} tool calls/question{key}"
    if not doc:
        return "Upload a PDF to begin. " + base
    return f"**{doc.title}** · {doc.n_pages} pages · `{doc.doc_id}`  \n{base}"


def on_upload(file, state):
    state = state or _new_state()
    if state.get("doc_id"):
        tools.unregister(state["doc_id"])
        state["doc_id"] = None
    if not file:
        return state, _doc_info(None), [], [], "", ""
    path = Path(file if isinstance(file, str) else file.name)
    try:
        doc = Document(path.read_bytes(), path.name)   # opened in memory; no text extracted here
    except Exception as exc:
        return state, f"Could not open that PDF: {exc}", [], [], "", ""
    tools.register(doc)
    state["doc_id"] = doc.doc_id
    state["session"].history.clear()
    return state, _doc_info(doc), [], [], "", ""


def _understanding_md(u: dict) -> str:
    parts = "; ".join(u["sub_parts"])
    return (f"**Question understood as:** {u['restated_question']}  \n"
            f"**Type:** {u['question_type']} · **Answer needs:** {u['answer_shape']}  \n"
            f"**Sub-parts:** {parts}  \n**Search terms:** {', '.join(u['search_terms'])}"
            + ("  \n**May be amended later in the document: yes**" if u.get("change_risk") else ""))


def on_ask(question, chat, state):
    chat = list(chat or [])
    rows: list[list] = []
    notes: list[str] = []
    if not question or not question.strip():
        yield chat, gr.update(), rows, gr.update(), ""
        return
    if not state or not state.get("doc_id"):
        chat += [{"role": "user", "content": question}, {"role": "assistant", "content": "Please upload a PDF first."}]
        yield chat, "", rows, "", ""
        return

    doc = tools.get_doc(state["doc_id"])
    chat += [{"role": "user", "content": question}, {"role": "assistant", "content": "Understanding the question…"}]
    und_md = ""
    yield chat, und_md, rows, "", ""
    try:
        for ev in AgentLoop(doc, state["session"]).run(question):
            kind = ev["type"]
            if kind == "understand":
                und_md = _understanding_md(ev["understanding"])
                chat[-1]["content"] = "Searching the document…"
            elif kind == "effort":
                e = ev["effort"]
                line = (f"**Reasoning effort:** planner {e['planner']} · reader {e['reader']} · "
                        f"answerer {e['answerer']} <sub>({ev['reason']})</sub>")
                und_md = und_md.split("\n**Reasoning effort:**")[0] + "  \n" + line if und_md else line
            elif kind == "step":
                status = ev["status"] if ev["status"] != "ok" else ""
                flags = ", ".join(([status] if status else []) + ev["flags"])
                args = ", ".join(f"{k}={v!r}" for k, v in ev["args"].items())
                rows.append([ev["step"], ev["tool"], args, ev["result_preview"], f"{ev['used']}/{ev['total']}",
                             flags, ev["reason"]])
            elif kind == "note":
                notes.append(f"- {ev['text']}")
            elif kind == "final":
                s = ev["stats"]
                footer = (f"\n\n<sub>Tool calls {s['tool_calls']}/{s['budget']} · LLM calls {s['llm_calls']} · "
                          f"{s['latency_s']} s · `{ev['question_id']}`</sub>")
                chat[-1]["content"] = ev["answer"] + footer
                if s["tool_calls"] >= s["budget"]:
                    notes.append("- **budget exhausted**")
            badges = _badges(rows, notes)
            yield chat, und_md, rows, badges, ""
    except Exception as exc:  # show errors visibly; never retry forever
        chat[-1]["content"] = f"⚠ Error: {exc}"
        notes.append(f"- **error:** {exc}")
        yield chat, und_md, rows, _badges(rows, notes), ""


def _badges(rows, notes) -> str:
    flags = " ".join(str(r[5]) for r in rows)
    badges = []
    if "instruction-like" in flags:
        badges.append("`instruction-like text detected (ignored)`")
    if "duplicate" in flags:
        badges.append("`duplicate rejected`")
    if "cut off" in flags:
        badges.append("`answer continues across pages`")
    if any("budget exhausted" in n for n in notes):
        badges.append("`budget exhausted`")
    return (" ".join(badges) + ("\n\n" if badges else "") + "\n".join(notes)).strip()


def on_reset(state):
    state = state or _new_state()
    state["session"] = Session()  # new session id and new delimiter; the PDF stays loaded
    return state, [], "", [], ""


def on_download(state):
    if not state:
        return None
    sid = state["session"].session_id
    events = calllog.read_events(session_id=sid)
    out = ROOT / "logs" / f"trace_{sid}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    return str(out)


with gr.Blocks(title="Budgeted Document Agent") as demo:
    gr.Markdown("## Budgeted document-answering agent\nNo RAG, no embeddings. At most "
                f"**{CONFIG.max_tool_calls} tool calls** per question; every call is shown and logged.")
    state = gr.State(_new_state())
    with gr.Row():
        with gr.Column(scale=5):
            pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            info = gr.Markdown(_doc_info(None))
            chat = _chatbot(label="Chat", height=480)
            question = gr.Textbox(placeholder="Ask a question about the PDF and press Enter", show_label=False)
            with gr.Row():
                reset = gr.Button("Reset session")
                dl_btn = gr.Button("Download trace")
            dl = gr.File(label="Session trace (JSONL)", interactive=False)
        with gr.Column(scale=6):
            und = gr.Markdown()
            trace = gr.Dataframe(headers=TRACE_HEADERS, value=[], interactive=False, label="Live tool trace")
            badges = gr.Markdown()

    pdf.change(on_upload, [pdf, state], [state, info, chat, trace, und, badges])
    question.submit(on_ask, [question, chat, state], [chat, und, trace, badges, question])
    reset.click(on_reset, [state], [state, chat, und, trace, badges])
    dl_btn.click(on_download, [state], [dl])


if __name__ == "__main__":
    demo.queue().launch(server_name=os.getenv("HOST", "127.0.0.1"), server_port=int(os.getenv("PORT", "7860")))
