"""
Report Generator Agent.

Turns validated research into a polished final report. Deliberately
split into two halves with different reliability requirements:

- `narrative_markdown` (Executive Summary, Key Findings, Detailed
  Findings/Analysis, Conclusion) is LLM-authored so its structure can
  adapt to the question — a comparison gets a table, a historical
  question gets a timeline, etc. — without us hand-coding section
  variants for every possible question shape.

- Everything else (sources, methodology, limitations, metadata) is
  built deterministically straight from ResearchResult/ReviewResult, so
  those parts can never contain a claim, url, or date that wasn't
  already validated earlier in the pipeline.

If the LLM call fails even after retries, a deterministic fallback
narrative is generated from the findings directly — the report always
gets produced, it just loses adaptive prose in the worst case.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict

from langchain_ollama import ChatOllama

from config.settings import settings
from models.schemas import FinalReport, ResearchPlan, ResearchResult, ReviewResult
from utils import logging as log
from utils.retry import call_with_retry
from utils.text import truncate

try:
    import markdown as _markdown_lib
except ImportError:  # pragma: no cover - optional dependency for HTML output
    _markdown_lib = None

_narrative_llm = None


def _get_narrative_llm():
    global _narrative_llm
    if _narrative_llm is None:
        _narrative_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
            num_predict=settings.narrative_max_output_tokens,
        )
    return _narrative_llm


SYSTEM_PROMPT = """
You are a research report writer. You will be given a research question,
its objectives, a Reviewer's quality assessment, and a list of validated
findings (each with a claim, confidence, evidence, source, and url).

Write the BODY of a research report in Markdown, containing exactly these
sections, in this order:

## Executive Summary
A short (3-6 sentence) summary of what the research found.

## Key Findings
A concise bulleted list of the most important, highest-confidence points.

## Detailed Findings & Analysis
Go through the findings in more depth. ADAPT THE STRUCTURE to the
question:
- If the question compares things, use a Markdown table comparing them
  across the relevant dimensions.
- If the question is historical/causal, present it as a chronological
  or cause-and-effect narrative.
- If the question is a general survey/overview, group findings by theme.
- Otherwise, use whatever structure best serves clarity for THIS
  specific question.
Cite sources inline using the source name in parentheses, e.g. "(IEA)".

## Conclusion
A short, direct answer to the original research question, and — if the
research was not fully approved — an honest note on what remains
uncertain.

STRICT RULES:
- Use ONLY the findings provided. Do not add facts, statistics, dates,
  or sources that are not in the findings list.
- Do not invent URLs or source names.
- If the findings don't fully cover some aspect of the question, say so
  plainly rather than filling the gap with assumption.
- Output ONLY the Markdown body described above — no preamble, no
  additional top-level heading for the whole report (that's added
  separately), no code fences.
"""


def _build_prompt(plan: ResearchPlan, research: ResearchResult, review: ReviewResult) -> str:
    findings_json = json.dumps(
        [f.model_dump() for f in research.findings],
        indent=2,
        ensure_ascii=False,
    )
    findings_json = truncate(findings_json, settings.max_context_chars)
    objectives_block = "\n".join(f"- {o}" for o in plan.objectives)

    return f"""
RESEARCH QUESTION:
{plan.research_question}

OBJECTIVES:
{objectives_block}

REVIEWER ASSESSMENT:
approved={review.approved}, score={review.score}/100
summary: {review.summary}
weaknesses: {review.weaknesses}

VALIDATED FINDINGS (JSON):
{findings_json}

Write the report body now, following the system instructions exactly.
"""


def _deterministic_fallback_narrative(plan: ResearchPlan, research: ResearchResult, review: ReviewResult) -> str:
    lines = [
        "## Executive Summary",
        "",
        f"This report synthesizes {len(research.findings)} finding(s) collected while researching: "
        f"\"{plan.research_question}\". Automated narrative generation was unavailable, so this section "
        "was assembled directly from the validated findings below.",
        "",
        "## Key Findings",
        "",
    ]
    for f in research.findings[:10]:
        lines.append(f"- **{f.claim}** (confidence: {f.confidence}, source: {f.source})")

    lines += ["", "## Detailed Findings & Analysis", ""]
    for f in research.findings:
        lines.append(f"- {f.claim}")
        lines.append(f"  - Evidence: {f.evidence}")
        lines.append(f"  - Relevance: {f.relevance}")
        lines.append(f"  - Source: [{f.source}]({f.url})")

    lines += [
        "",
        "## Conclusion",
        "",
        review.summary or "See findings above for the current state of the research.",
    ]
    return "\n".join(lines)


def _generate_narrative(plan: ResearchPlan, research: ResearchResult, review: ReviewResult) -> str:
    prompt = _build_prompt(plan, research, review)

    def _invoke():
        messages = [("system", SYSTEM_PROMPT), ("human", prompt)]
        response = _get_narrative_llm().invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        text = str(content).strip()
        if not text:
            raise ValueError("Narrative LLM returned empty content.")
        return text

    outcome = call_with_retry(
        _invoke,
        max_retries=settings.narrative_max_retries,
        base_delay=settings.retry_base_delay_seconds,
        max_delay=settings.retry_max_delay_seconds,
        on_retry=log.retrying,
    )

    if outcome.success:
        return outcome.value

    log.warn(f"Narrative generation failed [{outcome.kind}]: {outcome.error}. Using deterministic fallback.")
    return _deterministic_fallback_narrative(plan, research, review)


def _build_methodology(plan: ResearchPlan, research: ResearchResult) -> str:
    return (
        f"This research was conducted autonomously: a Planner decomposed the question into "
        f"{len(plan.tasks)} task(s), a Researcher agent ran web searches (via Tavily) for each task "
        f"and extracted structured findings, and a Reviewer agent scored the combined result across "
        f"relevance, evidence quality, source quality, completeness, consistency, specificity, "
        f"verification, and freshness — triggering additional refinement rounds when the score fell "
        f"short. In total, {research.searches_used} web search(es) were performed, yielding "
        f"{len(research.sources)} unique source(s) and {len(research.findings)} validated finding(s)."
    )


def _build_limitations(review: ReviewResult, research: ResearchResult) -> list[str]:
    limitations = list(review.weaknesses)
    if not review.approved:
        limitations.append(
            "This research did not meet the approval threshold within the configured number of rounds; "
            "treat findings below high confidence with appropriate caution."
        )
    if not research.findings:
        limitations.append("No findings were available at report generation time.")
    return limitations


def build_final_report(plan: ResearchPlan, research: ResearchResult, review: ReviewResult) -> FinalReport:
    log.section("REPORT GENERATOR")

    narrative = _generate_narrative(plan, research, review)

    report = FinalReport(
        research_question=plan.research_question,
        approved=review.approved,
        score=review.score,
        narrative_markdown=narrative,
        methodology=_build_methodology(plan, research),
        limitations=_build_limitations(review, research),
        findings=research.findings,
        sources=research.sources,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    log.success("Final report generated.")
    return report


# ============================================================
# Rendering / saving
# ============================================================

def _render_markdown(report: FinalReport) -> str:
    status = "✅ APPROVED" if report.approved else "⚠️ NOT FULLY APPROVED"

    sources_lines = []
    for s in report.sources:
        date_part = f" — {s.published_date}" if s.published_date else ""
        sources_lines.append(f"- [{s.title}]({s.url}){date_part} ({s.domain or 'unknown domain'})")

    limitations_lines = [f"- {item}" for item in report.limitations] or ["- None noted."]

    return "\n".join(
        [
            "# Research Report",
            "",
            f"**Research Question:** {report.research_question}",
            "",
            f"**Status:** {status}  |  **Score:** {report.score}/100  |  **Generated:** {report.generated_at}",
            "",
            "---",
            "",
            report.narrative_markdown,
            "",
            "---",
            "",
            "## Sources",
            "",
            *sources_lines,
            "",
            "## Research Methodology",
            "",
            report.methodology,
            "",
            "## Limitations",
            "",
            *limitations_lines,
            "",
        ]
    )


def _render_html(report: FinalReport, markdown_text: str) -> str:
    if _markdown_lib is not None:
        body = _markdown_lib.markdown(markdown_text, extensions=["tables"])
    else:
        # Minimal fallback if the optional `markdown` package isn't installed.
        body = "<pre>" + markdown_text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Research Report — {report.research_question}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.6; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def save_report(report: FinalReport, output_dir: str = None) -> Dict[str, str]:
    output_dir = output_dir or settings.reports_dir
    os.makedirs(output_dir, exist_ok=True)

    md_text = _render_markdown(report)
    md_path = os.path.join(output_dir, "final_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    json_path = os.path.join(output_dir, "final_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    html_path = os.path.join(output_dir, "final_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(report, md_text))

    log.success(f"Report saved to {md_path}, {json_path}, {html_path}")
    return {"markdown": md_path, "json": json_path, "html": html_path}
