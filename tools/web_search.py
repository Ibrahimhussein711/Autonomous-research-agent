"""
Web search tool backed by Tavily.

This is the ONLY search capability exposed to any agent in this project.
It intentionally does two things:

1. `search_web` — a LangChain @tool the Researcher's LLM can call. Returns
   a formatted string (what an LLM wants to read).
2. `search_web_raw` — the same search, returning structured SourceEvidence
   objects (what the extraction step wants to work with, and what lets us
   preserve raw evidence even if LLM extraction fails).

Both share `_run_tavily_search` so there's exactly one place that talks
to the Tavily API and exactly one place that handles its errors. The
Tavily client itself is created lazily so importing this module never
requires a real API key (needed for unit tests / mocking).
"""

from typing import List, Optional

from langchain_core.tools import tool
from tavily import TavilyClient

from config.settings import settings
from models.schemas import SourceEvidence
from utils.text import credibility_hint, domain_of, truncate

_client: Optional[TavilyClient] = None


class WebSearchError(Exception):
    """Raised when the Tavily search itself fails (not: zero results)."""


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not settings.tavily_api_key:
            raise WebSearchError("TAVILY_API_KEY is not set. Add it to your .env file.")
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def _run_tavily_search(query: str) -> List[SourceEvidence]:
    client = _get_client()

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=settings.tavily_max_results,
        )
    except WebSearchError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise WebSearchError(f"Tavily search failed for query '{query}': {exc}") from exc

    raw_results = response.get("results", []) if isinstance(response, dict) else []

    evidence: List[SourceEvidence] = []
    for result in raw_results:
        url = result.get("url", "") or ""
        evidence.append(
            SourceEvidence(
                title=result.get("title", "") or "Untitled source",
                url=url,
                domain=domain_of(url),
                content=truncate(result.get("content", "") or "", settings.max_search_result_chars),
                published_date=result.get("published_date") or None,
                credibility_hint=credibility_hint(url),
            )
        )

    return evidence


def search_web_raw(query: str) -> List[SourceEvidence]:
    """Search Tavily and return structured, traceable evidence."""
    return _run_tavily_search(query)


@tool
def search_web(query: str) -> str:
    """
    Search the web for recent, verifiable information relevant to a
    research query. Returns titles, URLs, and content snippets from
    real search results — never invent information not present here.
    """
    try:
        evidence = _run_tavily_search(query)
    except WebSearchError as exc:
        return f"SEARCH_ERROR: {exc}"

    if not evidence:
        return "No search results found for this query."

    formatted = []
    for item in evidence:
        date_line = f"\nPublished: {item.published_date}" if item.published_date else ""
        formatted.append(
            f"Title: {item.title}\nURL: {item.url}{date_line}\nContent:\n{item.content}"
        )

    return "\n---\n".join(formatted)
