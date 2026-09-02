"""
Researcher Agent.

Given ONE ResearchTask, decides what to search for, calls the Tavily
search tool (the only tool it has), and extracts structured Findings
from whatever it collects.

Three design points that matter here (see README "Error Handling"):

1. The search *strategy* is derived from `task.description` and
   `task.recommended_source_types` at call time — nothing about specific
   subject matter (energy, AI, finance...) is hardcoded into the prompt.

2. Evidence extraction is layered so a JSON-formatting failure can never
   silently erase real search results:
     a. try strict JSON-schema structured output
     b. if that fails, fall back to a plain JSON prompt + lenient parsing
        that keeps whatever findings *do* validate and drops the rest
     c. if that also fails, synthesize low-confidence findings directly
        from the raw (real, non-invented) search snippets so the round
        still produces something the Reviewer can react to

3. Search budget is capped both per-task (MAX_SEARCHES_PER_TASK) and
   across the whole run (MAX_TOTAL_SEARCHES, enforced by the caller
   passing in a smaller `max_searches` near the global limit).
"""

import json
from typing import List, Optional, Set

from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama

from config.settings import settings
from models.schemas import Finding, ResearchResult, ResearchTask, SourceEvidence
from tools.web_search import search_web, search_web_raw
from utils import logging as log
from utils.retry import call_with_retry
from utils.structured_output import extract_text, find_first_json_object
from utils.text import truncate

_llm_with_tools = None
_extractor_llm = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=settings.researcher_step_max_output_tokens,
        )
        # Note: unlike the Groq/OpenAI-style `parallel_tool_calls` option,
        # Ollama tool calling generally returns one tool call per turn
        # (support varies by model). The loop below already handles either
        # case — it iterates over however many tool_calls come back — so
        # this works correctly either way, just with one round-trip per
        # search rather than a batched one. That's fine locally: there's
        # no shared TPM budget to protect against, only wall-clock time.
        _llm_with_tools = llm.bind_tools([search_web])
    return _llm_with_tools


def _get_extractor_llm():
    global _extractor_llm
    if _extractor_llm is None:
        _extractor_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=settings.extraction_max_output_tokens,
        )
    return _extractor_llm


# ============================================================
# Prompt builders (dynamic — no topic-specific content)
# ============================================================

def _build_system_prompt(task: ResearchTask, max_searches: int) -> str:
    source_types = (
        ", ".join(task.recommended_source_types)
        if task.recommended_source_types
        else "the most authoritative sources available for this specific topic"
    )

    return f"""
You are an autonomous web research agent working on ONE research task.

TASK: {task.name}
DESCRIPTION: {task.description}

You have exactly ONE tool available: search_web. You must only call
search_web — never invent or call any other tool.

============================================================
RESEARCH STRATEGY
============================================================
Decide what to search for based ONLY on the task description above.
Break the task into distinct, complementary search queries that each
investigate a different angle of it (do not repeat the same query with
minor wording changes). Prefer source types such as: {source_types}.

If you can identify several distinct searches you need up front, you may
request more than one in the same turn if your tooling supports it —
otherwise one search per turn is fine.

Prefer recent, primary, and authoritative sources over aggregators or
opinion pieces. Do not invent information, URLs, or sources. Do not
answer from memory when web evidence is required.

============================================================
SEARCH LIMIT
============================================================
You may perform at most {max_searches} searches for this task. Stop as
soon as you have enough evidence to cover the task, even if you have
searches remaining.

============================================================
IMPORTANT
============================================================
You are responsible only for collecting web evidence. Do NOT write the
final report — a separate extraction step will turn your search results
into structured findings.
"""


def _build_extraction_prompt(task: ResearchTask, research_context: str) -> str:
    return f"""
You are a strict evidence extraction agent.

Extract verified findings ONLY from the supplied research material below.
Do NOT use outside knowledge. Do NOT invent facts, URLs, or sources.

============================================================
RESEARCH TASK
============================================================
{task.name}: {task.description}

============================================================
RESEARCH MATERIAL
============================================================
{research_context}

============================================================
REQUIREMENTS
============================================================
Extract the strongest findings that directly address the research task
above. Only include information explicitly supported by the research
material. If some aspect of the task isn't covered by the material,
simply omit it — do not invent it. Still return all valid findings for
the aspects that ARE covered.

Maximum findings: {settings.max_findings_per_task}.

Every finding must contain: claim, confidence (high/medium/low),
evidence, relevance, source, url. The url must be copied exactly from
the research material. The source must appear in the supplied material.

Respond with ONLY a JSON object of this exact shape, nothing else:
{{"task_id": "...", "findings": [{{"claim": "...", "confidence": "high|medium|low", "evidence": "...", "relevance": "...", "source": "...", "url": "..."}}], "summary": "..."}}
"""





def _validate_findings_leniently(raw_findings: list, max_findings: int) -> List[Finding]:
    """Keep whatever findings validate, drop the rest instead of failing everything."""
    valid: List[Finding] = []
    for item in raw_findings[:max_findings]:
        try:
            valid.append(Finding.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            log.warn(f"Dropping one malformed finding during lenient parsing: {exc}")
    return valid


def _salvage_findings_from_sources(sources: List[SourceEvidence], max_findings: int) -> List[Finding]:
    """
    Last-resort fallback: if the LLM never produced usable structured
    output, build low-confidence findings directly from the raw search
    snippets so real, non-invented evidence isn't thrown away.
    """
    salvaged = []
    for source in sources[:max_findings]:
        if not source.url:
            continue
        salvaged.append(
            Finding(
                claim=f"Raw search result (unextracted): {source.title}",
                confidence="low",
                evidence=truncate(source.content, 500),
                relevance="Collected for this task; automated extraction failed, "
                          "so this is the raw source material rather than a synthesized claim.",
                source=source.title or source.url,
                url=source.url,
            )
        )
    return salvaged


def _dedupe_sources(sources: List[SourceEvidence]) -> List[SourceEvidence]:
    seen: Set[str] = set()
    unique = []
    for s in sources:
        if s.url and s.url not in seen:
            seen.add(s.url)
            unique.append(s)
    return unique


# ============================================================
# Researcher
# ============================================================

def researcher_agent(
    task: ResearchTask,
    search_history: Set[str],
    max_searches: Optional[int] = None,
) -> ResearchResult:
    max_searches = settings.max_searches_per_task if max_searches is None else max(0, max_searches)

    log.section(f"RESEARCHER AGENT — {task.task_id}: {task.name}")

    if max_searches == 0:
        log.warn("Global search budget exhausted before this task could start.")
        return ResearchResult(task_id=task.task_id, findings=[], summary="Search budget exhausted.", searches_used=0)

    system_prompt = _build_system_prompt(task, max_searches)
    messages = [("system", system_prompt), ("human", task.description)]

    collected_sources: List[SourceEvidence] = []
    tool_transcript: List[str] = []
    search_count = 0

    def _invoke_llm():
        return _get_llm_with_tools().invoke(messages)

    outcome = call_with_retry(
        _invoke_llm,
        max_retries=settings.researcher_max_retries,
        base_delay=settings.retry_base_delay_seconds,
        max_delay=settings.retry_max_delay_seconds,
        on_retry=log.retrying,
    )
    if not outcome.success:
        log.error(f"Initial researcher call failed after {outcome.attempts} attempt(s) [{outcome.kind}]: {outcome.error}")
        return ResearchResult(
            task_id=task.task_id,
            findings=[],
            summary="Researcher failed before performing any search.",
            searches_used=0,
        )

    response = outcome.value
    messages.append(response)

    # ========================================================
    # Research loop
    # ========================================================
    while getattr(response, "tool_calls", None):
        if search_count >= max_searches:
            log.warn("Maximum search limit reached for this task.")
            break

        made_progress = False

        for tool_call in response.tool_calls:
            if tool_call["name"] != "search_web":
                log.warn(f"Unsupported tool call ignored: {tool_call['name']}")
                messages.append(
                    ToolMessage(
                        content="UNSUPPORTED_TOOL: only search_web is available.",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
                continue

            if search_count >= max_searches:
                # If a turn requests more searches than the remaining budget
                # (possible with models that batch tool calls), every
                # tool_call still needs a ToolMessage reply — most chat
                # APIs (Ollama included) require one response per call in
                # a batch, or the conversation becomes invalid for any
                # follow-up turn.
                messages.append(
                    ToolMessage(
                        content="SEARCH_LIMIT_REACHED: no searches remaining for this task.",
                        tool_call_id=tool_call["id"],
                        name="search_web",
                    )
                )
                continue

            query = str(tool_call["args"].get("query", "")).strip()
            normalized = query.lower()

            if normalized in search_history:
                log.warn(f"Skipping duplicate query already searched this session: '{query}'")
                messages.append(
                    ToolMessage(
                        content=(
                            "DUPLICATE_QUERY: this exact query was already searched "
                            "this session. Choose a different, more specific query."
                        ),
                        tool_call_id=tool_call["id"],
                        name="search_web",
                    )
                )
                continue

            log.info(f"🔎 Search {search_count + 1}/{max_searches}: {query}")

            def _do_search():
                return search_web_raw(query)

            search_outcome = call_with_retry(
                _do_search,
                max_retries=2,
                base_delay=settings.retry_base_delay_seconds,
                max_delay=settings.retry_max_delay_seconds,
                on_retry=log.retrying,
            )

            if not search_outcome.success:
                log.error(f"Search failed [{search_outcome.kind}]: {search_outcome.error}")
                messages.append(
                    ToolMessage(
                        content=f"SEARCH_ERROR: {search_outcome.error}",
                        tool_call_id=tool_call["id"],
                        name="search_web",
                    )
                )
                continue

            evidence = search_outcome.value
            search_history.add(normalized)
            search_count += 1
            made_progress = True
            collected_sources.extend(evidence)

            if evidence:
                # Full content (already capped at settings.max_search_result_chars
                # in tools/web_search.py) goes to the extraction transcript only —
                # extraction quality depends on it.
                full_formatted = "\n---\n".join(
                    f"Title: {e.title}\nURL: {e.url}\nContent:\n{e.content}" for e in evidence
                )
                # A much shorter version goes back into the LLM conversation.
                # The model only needs this to decide what to search next /
                # whether it has enough evidence — it doesn't need the full
                # snippet again, and resending full snippets on every
                # follow-up turn is what made the conversation grow O(n^2)
                # with each additional search.
                short_formatted = "\n---\n".join(
                    f"Title: {e.title}\nURL: {e.url}\nContent: "
                    f"{truncate(e.content, settings.max_search_snippet_chars_for_llm)}"
                    for e in evidence
                )
            else:
                full_formatted = "No search results found for this query."
                short_formatted = full_formatted

            tool_transcript.append(full_formatted)
            log.success(f"Search completed ({search_count}/{max_searches})")

            messages.append(
                ToolMessage(content=short_formatted, tool_call_id=tool_call["id"], name="search_web")
            )

        if search_count >= max_searches:
            log.warn("Maximum research searches reached.")
            break

        if not made_progress:
            log.warn("No progress made this iteration (all calls were duplicates or failed). Stopping.")
            break

        def _next_step():
            return _get_llm_with_tools().invoke(messages)

        next_outcome = call_with_retry(
            _next_step,
            max_retries=settings.researcher_max_retries,
            base_delay=settings.retry_base_delay_seconds,
            max_delay=settings.retry_max_delay_seconds,
            on_retry=log.retrying,
        )
        if not next_outcome.success:
            log.error(f"Research iteration failed after {next_outcome.attempts} attempt(s) [{next_outcome.kind}]: {next_outcome.error}")
            break

        response = next_outcome.value
        messages.append(response)

    collected_sources = _dedupe_sources(collected_sources)

    if not collected_sources:
        log.warn("No research material was collected for this task.")
        return ResearchResult(
            task_id=task.task_id,
            findings=[],
            summary="No research material was collected.",
            sources=[],
            searches_used=search_count,
        )

    # ========================================================
    # Extraction
    # ========================================================
    log.info("🧠 Extracting structured findings from collected evidence...")

    research_context = truncate("\n\n".join(tool_transcript), settings.max_context_chars)
    extraction_prompt = _build_extraction_prompt(task, research_context)

    findings = _extract_structured(extraction_prompt)

    if not findings:
        log.warn("Structured extraction produced nothing usable — salvaging raw evidence instead.")
        findings = _salvage_findings_from_sources(collected_sources, settings.max_findings_per_task)
        summary = (
            f"Automated extraction did not produce structured findings for '{task.name}'. "
            f"{len(findings)} raw source(s) are preserved below for manual review."
        )
    else:
        summary = f"Extracted {len(findings)} finding(s) for task '{task.name}'."

    log.success(
        f"Research complete for {task.task_id}: {len(findings)} finding(s), "
        f"{len(collected_sources)} source(s), {search_count} search(es) used."
    )

    return ResearchResult(
        task_id=task.task_id,
        findings=findings,
        summary=summary,
        sources=collected_sources,
        searches_used=search_count,
    )


def _extract_structured(extraction_prompt: str) -> List[Finding]:
    """
    Ask the extractor model for JSON matching the shape described in the
    prompt, then parse leniently: keep whatever individual findings DO
    validate and drop the rest, rather than failing the whole batch over
    one malformed item. (There's no Groq/OpenAI-style strict JSON-schema
    call here anymore -- that's an API feature Ollama doesn't expose the
    same way, and qwen2:latest doesn't reliably need it: plain "respond
    with only this JSON shape" prompting plus lenient parsing is both
    simpler and more portable across models.)

    If this produces nothing usable, the caller falls back further to
    salvaging raw evidence directly from the collected sources.
    """

    def _call():
        messages = [
            ("system", extraction_prompt),
            ("human", "Extract the verified findings now. Respond with JSON only, no markdown fences."),
        ]
        response = _get_extractor_llm().invoke(messages)
        text = extract_text(response)
        json_block = find_first_json_object(text)
        if json_block is None:
            raise ValueError("No JSON object found in extractor response.")
        return json.loads(json_block)

    # A couple of retries here are cheap and useful locally (guards against
    # a truncated/malformed generation), but if it keeps failing we don't
    # keep hammering -- the raw-evidence salvage path is a solid, free
    # fallback that never needs another LLM call.
    outcome = call_with_retry(
        _call,
        max_retries=2,
        base_delay=settings.retry_base_delay_seconds,
        max_delay=settings.retry_max_delay_seconds,
        on_retry=log.retrying,
    )
    if not outcome.success:
        log.error(f"Extraction failed [{outcome.kind}]: {outcome.error}")
        return []

    raw_findings = outcome.value.get("findings", []) if isinstance(outcome.value, dict) else []
    return _validate_findings_leniently(raw_findings, settings.max_findings_per_task)
