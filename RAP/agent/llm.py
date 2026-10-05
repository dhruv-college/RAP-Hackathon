"""LLM client: one OpenAI-compatible client for Groq (default), OpenAI, Gemini or Anthropic.

- call_llm(role, messages, schema=None)  -> str, or a validated pydantic object when schema is given
- call_tools(role, messages, tools)      -> (tool_name, arguments, text) via NATIVE tool calling

Structured outputs: strict JSON schema first (Groq gpt-oss supports strict mode), falling back to
json_object, then to prompt-only JSON if the provider rejects a mode. Parse/validation failures get
ONE corrective retry, then LLMFormatError. 429/5xx/network errors back off (max 3 attempts,
Retry-After respected). LLM retries never consume tool budget. Every call is logged.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ValidationError

from . import calllog
from .config import API_KEY_VARS, CONFIG
from .schemas import STRICT_SCHEMAS


class LLMError(RuntimeError):
    pass


class LLMFormatError(LLMError):
    pass


@dataclass
class Msg:
    content: str = ""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, arguments json string)
    reasoning: str = ""


# Tests plug in a scripted fake: fn(role, messages, mode, payload) -> dict | str
#   mode "json"  -> dict (or str) for the schema named in payload
#   mode "tools" -> {"name": ..., "arguments": {...}} or {"content": "..."}
#   mode "text"  -> str
_fake: Callable | None = None
_clients: dict = {}
_downgrades: dict = {}  # (provider, model) -> response_format modes the provider rejected


def set_fake(fn: Callable | None) -> None:
    global _fake
    _fake = fn


def describe() -> str:
    if _fake:
        return "scripted fake LLM"
    return f"{CONFIG.llm_provider}: strong={CONFIG.model_strong}, fast={CONFIG.model_fast}"


def role_settings(role: str, ctx: dict | None = None) -> tuple[str, str]:
    model = CONFIG.model_fast if role == "reader" else CONFIG.model_strong
    effort = {
        "understand": CONFIG.reasoning_effort_understand,
        "planner": CONFIG.reasoning_effort_planner,
        "reader": CONFIG.reasoning_effort_reader,
        "answerer": CONFIG.reasoning_effort_answerer,
    }.get(role, "")
    effort = ((ctx or {}).get("effort") or {}).get(role, effort)  # per-question choice from agent/effort.py
    if not model and not _fake:
        raise LLMError(f"Set MODEL_STRONG and MODEL_FAST for provider '{CONFIG.llm_provider}'.")
    return model, effort


def _client():
    key = (CONFIG.llm_provider, CONFIG.base_url, CONFIG.api_key)
    if key not in _clients:
        if not CONFIG.api_key:
            var = API_KEY_VARS.get(CONFIG.llm_provider, "API key")
            raise LLMError(f"{var} is not set. Put it in a .env file in the project folder ({var}=...).")
        from openai import OpenAI
        _clients[key] = OpenAI(api_key=CONFIG.api_key, base_url=CONFIG.base_url,
                               max_retries=0, timeout=CONFIG.llm_timeout_s)
    return _clients[key]


def _log(ctx, role, model, mode, status, started, msg: Msg | None = None, usage=None, error=None):
    if ctx is not None:
        ctx["llm_calls"] = ctx.get("llm_calls", 0) + 1
    out = None
    if msg is not None:
        out = msg.content or ""
        if msg.tool_calls:
            out = "; ".join(f"{n}({a})" for n, a in msg.tool_calls) + (f" | {out}" if out else "")
    calllog.get_logger().log({
        "session_id": (ctx or {}).get("session_id", ""), "question_id": (ctx or {}).get("question_id", ""),
        "kind": "llm", "name": role, "model": model, "mode": mode, "args": {},
        "reasoning_effort": (ctx or {}).get("_last_effort"),
        "result_preview": calllog.preview(out, 600) if out is not None else None,
        "reasoning_preview": calllog.preview(msg.reasoning, 400) if msg and msg.reasoning else None,
        "tokens": usage, "duration_ms": round((time.time() - started) * 1000, 1),
        "budget_used": None, "status": status, "error": error,
    })


def _status_code(exc) -> int | None:
    return getattr(exc, "status_code", None)


def _retry_after(exc) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value else None
    except Exception:
        return None


def _create(role: str, messages: list[dict], *, ctx=None, mode: str = "text",
            schema_name: str | None = None, tools: list | None = None) -> Msg:
    model, effort = role_settings(role, ctx)
    started = time.time()
    if ctx is not None:
        ctx["_last_effort"] = effort

    if _fake:
        out = _fake(role, messages, mode, schema_name or tools)
        if mode == "tools":
            out = out or {}
            msg = Msg(content=out.get("content", "") or "",
                      tool_calls=[(out["name"], json.dumps(out.get("arguments", {})))] if out.get("name") else [])
        else:
            msg = Msg(content=out if isinstance(out, str) else json.dumps(out))
        _log(ctx, role, "fake", mode, "ok", started, msg)
        return msg

    client = _client()
    use_effort = bool(effort)
    formats = ["json_schema", "json_object", "none"] if mode == "json" else ["none"]
    formats = [f for f in formats if f not in _downgrades.get((CONFIG.llm_provider, model), set())] or ["none"]
    tool_choice = "required"
    attempt = 0
    while True:
        kwargs = dict(model=model, messages=messages, temperature=CONFIG.temperature)
        if use_effort:
            kwargs["extra_body"] = {"reasoning_effort": effort}
        fmt = formats[0]
        if fmt == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema_name, "schema": STRICT_SCHEMAS[schema_name], "strict": True}}
        elif fmt == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if mode == "tools":
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            code, text = _status_code(exc), str(exc).lower()
            if code == 400 and use_effort and "reasoning" in text:
                use_effort = False                       # provider rejects reasoning_effort: drop it once
                continue
            if code == 400 and fmt != "none" and any(w in text for w in ("response_format", "json_schema", "schema", "json mode")):
                _downgrades.setdefault((CONFIG.llm_provider, model), set()).add(fmt)
                formats = formats[1:] or ["none"]        # step down: strict schema -> json_object -> none
                continue
            if code == 400 and mode == "tools" and tool_choice == "required" and "tool_choice" in text:
                tool_choice = "auto"
                continue
            retryable = code in (408, 409, 429) or (code is not None and code >= 500) or code is None
            attempt += 1
            if retryable and attempt < 3:
                time.sleep(_retry_after(exc) or (2 ** attempt + random.random()))
                continue
            _log(ctx, role, model, mode, "error", started, error=str(exc)[:500])
            raise LLMError(f"{role} call failed: {exc}") from exc

        m = resp.choices[0].message
        msg = Msg(content=m.content or "",
                  tool_calls=[(tc.function.name, tc.function.arguments or "{}") for tc in (m.tool_calls or [])],
                  reasoning=getattr(m, "reasoning", "") or "")
        usage = None
        if getattr(resp, "usage", None):
            usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens}
        _log(ctx, role, model, mode, "ok", started, msg, usage)
        return msg


def extract_json(text: str) -> dict:
    """First JSON object in a reply (tolerates code fences and leading prose)."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object")


def call_llm(role: str, messages: list[dict], schema: type[BaseModel] | None = None, *, ctx=None):
    if schema is None:
        return _create(role, messages, ctx=ctx, mode="text").content
    msgs = list(messages)
    last_err = None
    for _ in range(2):  # first try + ONE corrective retry
        msg = _create(role, msgs, ctx=ctx, mode="json", schema_name=schema.__name__)
        try:
            return schema.model_validate(extract_json(msg.content))
        except (ValueError, ValidationError) as exc:
            last_err = exc
            msgs = msgs + [
                {"role": "assistant", "content": msg.content or "(empty)"},
                {"role": "user", "content": f"That reply did not match the required JSON schema: {str(exc)[:400]}. "
                                            "Reply with ONLY the corrected JSON object."},
            ]
    raise LLMFormatError(f"{role}: invalid structured output after retry: {last_err}")


def call_tools(role: str, messages: list[dict], tools: list[dict], *, ctx=None) -> tuple[str | None, dict, str]:
    """Native tool calling. Returns the FIRST tool call as (name, arguments, text)."""
    msg = _create(role, messages, ctx=ctx, mode="tools", tools=tools)
    if not msg.tool_calls:
        return None, {}, msg.content
    name, raw_args = msg.tool_calls[0]
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments are not an object")
    except ValueError:
        args = {"__invalid_json__": raw_args}
    return name, args, msg.content
