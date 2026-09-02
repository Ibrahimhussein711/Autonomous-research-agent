"""
Startup diagnostic: verify the local Ollama server is reachable and that
the configured model is actually pulled, BEFORE running the full pipeline.
Uses only the standard library (urllib) so it doesn't need an extra
dependency just for a health check.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Tuple

from config.settings import settings


def check_ollama_available(timeout: float = 4.0) -> Tuple[bool, str]:
    """Returns (ok, message)."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, (
            f"Cannot reach Ollama at {settings.ollama_base_url} ({exc.reason if hasattr(exc, 'reason') else exc}). "
            f"Is `ollama serve` running?"
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error contacting Ollama at {settings.ollama_base_url}: {exc}"

    models = [m.get("name", "") for m in data.get("models", [])]
    target = settings.ollama_model
    target_base = target.split(":")[0]

    if target in models or any(m.split(":")[0] == target_base for m in models):
        return True, f"Ollama reachable at {settings.ollama_base_url}; model '{target}' is available."

    available = ", ".join(models) if models else "(none)"
    return False, (
        f"Ollama is reachable, but model '{target}' was not found. "
        f"Available models: {available}. Run `ollama pull {target}` if needed."
    )
