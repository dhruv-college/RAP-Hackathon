"""Configuration from environment variables (and an optional .env file in the project root).

A .env file avoids shell differences (PowerShell vs bash): put `GROQ_API_KEY=gsk_...` in it.
Real environment variables win over .env values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _bool(name: str, default: str) -> bool:
    return _env(name, default).strip().lower() in ("1", "true", "yes", "on")


# OpenAI-compatible base URLs. Check each provider's docs before relying on them.
BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": None,  # SDK default
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1/",
}
API_KEY_VARS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass
class Config:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    # Model names default only for Groq; other providers must set them explicitly.
    model_strong: str = field(default_factory=lambda: _env(
        "MODEL_STRONG", "openai/gpt-oss-120b" if _env("LLM_PROVIDER", "groq").lower() == "groq" else ""))
    model_fast: str = field(default_factory=lambda: _env(
        "MODEL_FAST", "openai/gpt-oss-20b" if _env("LLM_PROVIDER", "groq").lower() == "groq" else ""))
    max_tool_calls: int = field(default_factory=lambda: int(_env("MAX_TOOL_CALLS", "6")))
    max_planner_steps: int = field(default_factory=lambda: int(_env("MAX_PLANNER_STEPS", "10")))
    max_page_chars: int = field(default_factory=lambda: int(_env("MAX_PAGE_CHARS", "12000")))
    max_headings: int = field(default_factory=lambda: int(_env("MAX_HEADINGS", "200")))
    use_reader: bool = field(default_factory=lambda: _bool("USE_READER", "true"))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0")))
    # Adaptive effort (agent/effort.py) picks planner/reader/answerer effort per question.
    # The fixed REASONING_EFFORT_* values below apply when ADAPTIVE_EFFORT=false.
    adaptive_effort: bool = field(default_factory=lambda: _bool("ADAPTIVE_EFFORT", "true"))
    reasoning_effort_ceiling: str = field(default_factory=lambda: _env("REASONING_EFFORT_CEILING", "high"))
    reasoning_effort_understand: str = field(default_factory=lambda: _env("REASONING_EFFORT_UNDERSTAND", "low"))
    reasoning_effort_planner: str = field(default_factory=lambda: _env("REASONING_EFFORT_PLANNER", "medium"))
    reasoning_effort_reader: str = field(default_factory=lambda: _env("REASONING_EFFORT_READER", "low"))
    reasoning_effort_answerer: str = field(default_factory=lambda: _env("REASONING_EFFORT_ANSWERER", "medium"))
    max_quotes_per_page: int = 3
    max_quote_chars: int = 400
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "60")))
    log_path: Path = field(default_factory=lambda: Path(_env("CALL_LOG_PATH", str(ROOT / "logs" / "calls.jsonl"))))

    @property
    def api_key(self) -> str:
        return os.getenv(API_KEY_VARS.get(self.llm_provider, ""), "")

    @property
    def base_url(self) -> str | None:
        return BASE_URLS.get(self.llm_provider)


CONFIG = Config()

INSUFFICIENT_MESSAGE = "Insufficient information: the document does not contain enough to answer this."
