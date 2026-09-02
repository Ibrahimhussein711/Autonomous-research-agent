"""
Provider-agnostic structured-output helper with a manual JSON fallback.

The project previously relied on `.with_structured_output(..., method="json_schema")`
against Groq, which is a strict-schema feature of Groq/OpenAI-compatible APIs.
Ollama has its own native structured-output support (also exposed via
`method="json_schema"` in recent langchain-ollama versions), but its
reliability depends on the installed Ollama server version AND on how well
the specific local model (e.g. `qwen2:latest`, an older/smaller model)
follows a JSON schema.

`invoke_structured()` always tries the native path first and, if it fails
for ANY reason (unsupported method, malformed output, validation error),
falls back to a manual "describe the JSON shape in the prompt, then parse
leniently" approach — the same pattern already used for evidence
extraction in agents/researcher.py. Tool calling and structured extraction
stay fully separate call sites, so this never re-introduces the old
tool-calling + structured-output-in-one-call problem.
"""

from __future__ import annotations

import json
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from utils import logging as log

T = TypeVar("T", bound=BaseModel)


def find_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced {...} block from arbitrary text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def invoke_structured(raw_llm, messages: list, schema_cls: Type[T], json_shape_hint: str) -> T:
    """
    Try native structured output first; fall back to a manual JSON prompt +
    lenient parsing if that fails for any reason. Raises the last error if
    both approaches fail, so callers can apply their own fallback behavior
    (the Planner/Reviewer already have their own graceful fallbacks for
    when this raises).
    """
    # --- Attempt 1: native structured output --------------------------
    try:
        structured_llm = raw_llm.with_structured_output(schema_cls)
        result = structured_llm.invoke(messages)
        if isinstance(result, schema_cls):
            return result
        if isinstance(result, dict):
            return schema_cls.model_validate(result)
        raise TypeError(f"Unexpected structured output type: {type(result)}")
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure falls through
        log.warn(f"Native structured output failed ({exc}); falling back to manual JSON parsing.")

    # --- Attempt 2: manual JSON prompt + lenient parsing ---------------
    fallback_messages = list(messages) + [
        (
            "human",
            f"Respond with ONLY a single JSON object of this exact shape, nothing else, "
            f"no markdown code fences:\n{json_shape_hint}",
        )
    ]
    response = raw_llm.invoke(fallback_messages)
    text = extract_text(response)
    json_block = find_first_json_object(text)
    if json_block is None:
        raise ValueError(f"No JSON object found in fallback response: {text[:200]!r}")
    data = json.loads(json_block)
    return schema_cls.model_validate(data)
