"""
Reviewer Agent.

Evaluates a (merged) ResearchResult against the original question and
ResearchPlan. Never searches the web, never adds facts. Its
`recommendations` are what drive the next research round — the
Orchestrator turns each recommendation directly into a new ResearchTask.
"""

from config.settings import settings
from models.schemas import ResearchPlan, ResearchResult, ReviewResult
from utils import logging as log
from utils.retry import call_with_retry
from utils.structured_output import invoke_structured
from utils.text import truncate
from langchain_ollama import ChatOllama

_raw_llm = None


def _get_raw_llm():
    global _raw_llm
    if _raw_llm is None:
        _raw_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=settings.reviewer_max_output_tokens,
        )
    return _raw_llm


SYSTEM_PROMPT = """
You are a strict research quality reviewer.

Evaluate the provided research result against the original research
question and its objectives. Do NOT perform web searches, add new facts,
or use outside knowledge.

Evaluate across these dimensions:
1. Relevance — does the research answer the question?
2. Evidence quality — are claims backed by concrete evidence?
3. Source quality — are sources credible and appropriate for the topic?
4. Completeness — do the findings cover the plan's stated objectives?
5. Consistency — are there contradictions between findings?
6. Specificity — are claims concrete rather than vague?
7. Verification — can every claim be connected to a real source/url?
8. Freshness — if the question needs current information, is it recent?

Scoring:
90-100 = excellent, 75-89 = good, 60-74 = needs improvement, 0-59 = insufficient.

Approval rule — approved = true ONLY when ALL of the following hold:
- score >= 75
- the research is clearly relevant to the question
- evidence is sufficient to support the main claims
- there are no major unsupported claims

If rejected, `recommendations` MUST be specific and actionable — each one
should read like an instruction for what to research next (e.g. "Find
recent official statistics on X from a government or standards body"),
not a vague restatement of a weakness. Each recommendation becomes a new,
standalone research task, so make it self-contained.

Be concise. Return only a ReviewResult.
"""

_REVIEW_JSON_SHAPE = """{
  "approved": true,
  "score": 85,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "recommendations": ["...", "..."],
  "summary": "..."
}"""


def reviewer_agent(
    question: str,
    plan: ResearchPlan,
    research: ResearchResult,
) -> ReviewResult:
    log.section("REVIEWER AGENT")

    if not research.findings:
        log.warn("No findings to review.")
        return ReviewResult(
            approved=False,
            score=0,
            strengths=[],
            weaknesses=["No verified research findings were produced."],
            recommendations=["Collect additional verified web evidence covering the plan's objectives."],
            summary="The research result is empty and cannot be approved.",
        )

    objectives_block = "\n".join(f"- {o}" for o in plan.objectives)
    research_json = truncate(
        research.model_dump_json(indent=2, exclude={"sources"}),
        settings.max_context_chars,
    )

    review_prompt = f"""
ORIGINAL RESEARCH QUESTION:
{question}

RESEARCH OBJECTIVES:
{objectives_block}

RESEARCH RESULT TO REVIEW:
{research_json}

Review the research strictly according to the system instructions.
Do not invent missing evidence. Return a ReviewResult.
"""

    def _invoke():
        messages = [("system", SYSTEM_PROMPT), ("human", review_prompt)]
        return invoke_structured(_get_raw_llm(), messages, ReviewResult, _REVIEW_JSON_SHAPE)

    outcome = call_with_retry(
        _invoke,
        max_retries=settings.reviewer_max_retries,
        base_delay=settings.retry_base_delay_seconds,
        max_delay=settings.retry_max_delay_seconds,
        on_retry=log.retrying,
    )

    if outcome.success:
        log.success(f"Review completed — score {outcome.value.score}/100, approved={outcome.value.approved}")
        return outcome.value

    log.error(f"Reviewer failed after {outcome.attempts} attempt(s) [{outcome.kind}]: {outcome.error}")
    return ReviewResult(
        approved=False,
        score=0,
        strengths=[],
        weaknesses=["Reviewer execution failed."],
        recommendations=["Retry the research after the underlying API issue resolves."],
        summary="The research could not be reviewed because the reviewer model failed.",
    )
