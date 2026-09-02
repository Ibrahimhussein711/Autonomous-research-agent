"""
Small, dependency-free retry helper with exponential backoff.

This project makes calls to a local Ollama server and to Tavily, both of
which can fail transiently (Ollama: connection refused while starting up
or loading a model into memory, timeouts; Tavily: network errors). We
never retry forever, we distinguish *why* a call failed so the caller
can log something useful, and — for any provider that reports a
server-suggested wait time in its error message (a pattern originally
seen from Groq, kept here since it's harmless and still generically
useful) — we respect that instead of guessing.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

_RETRY_AFTER_PATTERN = re.compile(r"try again in\s+([\d.]+)\s*s", re.IGNORECASE)


class FailureKind:
    """Coarse classification of why an external call failed."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    MODEL_NOT_FOUND = "model_not_found"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


# Kinds that are worth retrying at all. Authentication and model-not-found
# are permanent — retrying with the same key/model can't fix them.
_RETRYABLE = {FailureKind.RATE_LIMIT, FailureKind.TIMEOUT, FailureKind.CONNECTION}


def classify_error(error: Exception) -> str:
    text = str(error).lower()

    if "429" in text or "rate limit" in text or "too many requests" in text or "tokens per minute" in text or "tpm" in text:
        return FailureKind.RATE_LIMIT

    if "401" in text or "403" in text or "invalid api key" in text or "authentication" in text or "unauthorized" in text:
        return FailureKind.AUTHENTICATION

    if "model_not_found" in text or ("model" in text and "does not exist" in text) or "decommissioned" in text:
        return FailureKind.MODEL_NOT_FOUND

    if "timeout" in text or "timed out" in text:
        return FailureKind.TIMEOUT

    if "connection" in text:
        return FailureKind.CONNECTION

    if "validation" in text or "json_validate_failed" in text or "schema" in text:
        return FailureKind.VALIDATION

    return FailureKind.UNKNOWN


def is_retryable(kind: str) -> bool:
    return kind in _RETRYABLE


def extract_retry_after_seconds(error: Exception) -> Optional[float]:
    """
    Groq (and most OpenAI-compatible APIs) put the server-suggested wait
    time directly in the error message, e.g. '...Please try again in
    1.234s'. Prefer that over guessing with exponential backoff when
    it's available.
    """
    match = _RETRY_AFTER_PATTERN.search(str(error))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


@dataclass
class RetryOutcome:
    success: bool
    value: object = None
    error: Exception | None = None
    kind: str | None = None
    attempts: int = 0


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    on_retry: Callable[[int, Exception, str, float], None] | None = None,
) -> RetryOutcome:
    """
    Call `fn()` with limited retries and exponential backoff.

    Only retries errors classified as transient (rate limit, timeout,
    connection). Permanent errors (auth, model-not-found, validation)
    fail immediately since retrying identical input can't fix them.

    For rate limits, uses the server's own "try again in Xs" hint when
    present instead of blind exponential backoff.

    Returns a RetryOutcome instead of raising, so callers can decide how
    to degrade gracefully (fallback value, partial result, etc.) without
    a try/except at every call site.
    """

    last_error: Exception | None = None
    last_kind: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            value = fn()
            return RetryOutcome(success=True, value=value, attempts=attempt)
        except Exception as error:  # noqa: BLE001 - this is a generic retry wrapper
            kind = classify_error(error)
            last_error = error
            last_kind = kind

            if not is_retryable(kind) or attempt == max_retries:
                return RetryOutcome(success=False, error=error, kind=kind, attempts=attempt)

            server_delay = extract_retry_after_seconds(error) if kind == FailureKind.RATE_LIMIT else None
            if server_delay is not None:
                # Groq told us exactly how long to wait — use it as-is, no jitter needed.
                delay = server_delay
            else:
                # No server hint: exponential backoff + jitter, so concurrent
                # retries (e.g. across tasks) don't all wake up and re-hit
                # the API in the same instant.
                exp_delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                delay = exp_delay * random.uniform(0.7, 1.3)

            delay = min(delay, max_delay)

            if on_retry:
                on_retry(attempt, error, kind, delay)

            time.sleep(delay)

    return RetryOutcome(success=False, error=last_error, kind=last_kind, attempts=max_retries)
