"""
Centralized configuration.

Every agent reads its model name and limits from `settings` in this
module instead of calling `os.getenv` directly. That was the biggest
smell in the original codebase — the model name was hardcoded
separately in three different agent files, so changing it meant
editing three places and inevitably missing one.

Nothing here raises on missing config at import time — that would break
unit tests and any tooling that imports agents without running them.
`main.py` is responsible for checking `settings.validate()` and Ollama
reachability at startup and failing loudly there, before any agent runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- API keys -----------------------------------------------------
    # Groq is no longer used anywhere in the active runtime — the LLM is
    # now local via Ollama, so there's no API key for it.
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # --- Model (single source of truth for every agent) -----------------
    # Local Ollama model. Deliberately NOT auto-selecting a "better"
    # remote model — this project runs entirely against whatever is
    # already pulled locally.
    ollama_model: str = field(default_factory=lambda: _env_str("OLLAMA_MODEL", "qwen2:latest"))
    ollama_base_url: str = field(default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434"))

    # --- Research limits ----------------------------------------------
    max_searches_per_task: int = field(default_factory=lambda: _env_int("MAX_SEARCHES_PER_TASK", 3))
    max_research_rounds: int = field(default_factory=lambda: _env_int("MAX_RESEARCH_ROUNDS", 3))
    max_total_searches: int = field(default_factory=lambda: _env_int("MAX_TOTAL_SEARCHES", 20))
    max_findings_per_task: int = field(default_factory=lambda: _env_int("MAX_FINDINGS", 8))

    # --- Output length control (provider-agnostic) -----------------------
    # These map to Ollama's native `num_predict` option per call. Kept
    # per-call-type (rather than one blanket value) because a small
    # tool-call-decision step needs far fewer tokens than a full report
    # narrative, and local generation speed benefits from not over-asking.
    max_output_tokens: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", 1200))
    planner_max_output_tokens: int = field(default_factory=lambda: _env_int("PLANNER_MAX_OUTPUT_TOKENS", 700))
    researcher_step_max_output_tokens: int = field(
        default_factory=lambda: _env_int("RESEARCHER_STEP_MAX_OUTPUT_TOKENS", 300)
    )
    extraction_max_output_tokens: int = field(default_factory=lambda: _env_int("EXTRACTION_MAX_OUTPUT_TOKENS", 900))
    reviewer_max_output_tokens: int = field(default_factory=lambda: _env_int("REVIEWER_MAX_OUTPUT_TOKENS", 600))
    narrative_max_output_tokens: int = field(default_factory=lambda: _env_int("NARRATIVE_MAX_OUTPUT_TOKENS", 1000))

    # Full snippet length used for SourceEvidence.content (feeds extraction
    # quality) vs. the much shorter version echoed back into the
    # tool-calling conversation (feeds only the next search decision).
    # Splitting these two keeps the conversation compact regardless of
    # provider — still useful locally since a smaller local model has a
    # smaller reliable context window than a hosted one.
    max_search_result_chars: int = field(default_factory=lambda: _env_int("MAX_SEARCH_RESULT_CHARS", 1000))
    max_search_snippet_chars_for_llm: int = field(
        default_factory=lambda: _env_int("MAX_SEARCH_SNIPPET_CHARS_FOR_LLM", 220)
    )
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 9000))

    # --- Retry / error handling ------------------------------------------
    # No TPM/rate-limit concerns with a local server, but Ollama can still
    # be briefly unavailable (e.g. still loading a model into memory) or
    # time out on a slow local machine, so bounded retries are still worth
    # keeping — just no longer Groq-specific.
    planner_max_retries: int = field(default_factory=lambda: _env_int("PLANNER_MAX_RETRIES", 3))
    researcher_max_retries: int = field(default_factory=lambda: _env_int("RESEARCHER_MAX_RETRIES", 3))
    reviewer_max_retries: int = field(default_factory=lambda: _env_int("REVIEWER_MAX_RETRIES", 3))
    narrative_max_retries: int = field(default_factory=lambda: _env_int("NARRATIVE_MAX_RETRIES", 2))
    retry_base_delay_seconds: float = field(default_factory=lambda: _env_float("RETRY_BASE_DELAY", 2.0))
    retry_max_delay_seconds: float = field(default_factory=lambda: _env_float("RETRY_MAX_DELAY", 30.0))

    # --- Tavily -----------------------------------------------------
    tavily_max_results: int = field(default_factory=lambda: _env_int("TAVILY_MAX_RESULTS", 3))

    # --- Output -------------------------------------------------------
    reports_dir: str = field(default_factory=lambda: _env_str("REPORTS_DIR", "reports"))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems, empty if configuration is OK.

        Ollama reachability is checked separately (utils/ollama_check.py)
        since it requires a live network call, not just reading env vars.
        """
        problems = []
        if not self.tavily_api_key:
            problems.append("TAVILY_API_KEY is missing. Add it to .env.")
        return problems


settings = Settings()
