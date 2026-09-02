"""
Small text utilities used to keep prompts and context compact — matters
for staying within a local model's reliable context window, and for
generation speed.
"""

from __future__ import annotations

from urllib.parse import urlparse


def truncate(text: str, max_chars: int, marker: str = " …[truncated]") -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    cut = max(0, max_chars - len(marker))
    return text[:cut] + marker


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for budgeting, not billing."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _message_content_to_text(message) -> str:
    """Extract raw text length from a LangChain-style message (tuple or BaseMessage)."""
    if isinstance(message, tuple):
        content = message[1]
    else:
        content = getattr(message, "content", "")

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def estimate_messages_tokens(messages, extra_reserved_tokens: int = 0) -> int:
    """
    Estimate the total input tokens a chat call will use, given a list of
    messages (tuples or BaseMessage objects) — used to proactively check a
    call against the TPM budget before it's sent, not just to react to a
    429 after the fact.
    """
    total_chars = sum(len(_message_content_to_text(m)) for m in messages)
    return max(1, total_chars // 4) + extra_reserved_tokens


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc.lower().removeprefix("www.")
    except Exception:  # noqa: BLE001
        return ""


# Cross-topic authoritative domain patterns — this is about *source type*
# (government, intergovernmental, standards, academic), not any specific
# research subject, so it stays general-purpose across arbitrary questions.
_HIGH_CREDIBILITY_SUFFIXES = (".gov", ".mil", ".int", ".edu")
_HIGH_CREDIBILITY_DOMAINS = {
    "un.org", "who.int", "worldbank.org", "imf.org", "oecd.org",
    "nature.com", "science.org", "ieee.org", "acm.org", "arxiv.org",
    "nist.gov", "europa.eu", "un.int",
}


def credibility_hint(url: str) -> str:
    """A soft, general-purpose heuristic — never used to silently drop a source,
    only to help the Researcher/Reviewer weigh evidence."""
    domain = domain_of(url)
    if not domain:
        return "unknown"
    if domain in _HIGH_CREDIBILITY_DOMAINS:
        return "high"
    if any(domain.endswith(suffix) for suffix in _HIGH_CREDIBILITY_SUFFIXES):
        return "high"
    return "unknown"
