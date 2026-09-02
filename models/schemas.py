"""
Pydantic schemas shared across all agents.

Keep this file boring on purpose: every agent in this project imports
from here, so the schemas are the contract between them. If you change
a field name, grep the whole repo before you commit.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# Research Plan (produced by the Planner)
# ============================================================

class ResearchTask(BaseModel):
    """A single, self-contained unit of research the Researcher can act on."""

    task_id: str = Field(
        description="Unique identifier for the research task, e.g. 'T1'."
    )

    name: str = Field(
        description="Short human-readable name of the task."
    )

    description: str = Field(
        description="Clear, specific instruction describing what should be researched."
    )

    recommended_source_types: List[str] = Field(
        default_factory=list,
        description="Types of sources likely to have good evidence for this task "
                    "(e.g. 'peer-reviewed research', 'government statistics', "
                    "'official vendor documentation'). Not hardcoded per topic — "
                    "the Planner decides these per question.",
    )


class ResearchPlan(BaseModel):
    """The Planner's dynamic breakdown of a research question."""

    research_question: str = Field(
        description="The original research question, verbatim."
    )

    objectives: List[str] = Field(
        description="The main things the research must establish to answer the question."
    )

    tasks: List[ResearchTask] = Field(
        description="Actionable, complementary research tasks that together cover the objectives."
    )


# ============================================================
# Evidence & Findings (produced by the Researcher)
# ============================================================

class SourceEvidence(BaseModel):
    """Raw evidence pulled from a single web search result.

    This is kept separate from `Finding` so we always have an unmodified
    record of what the search tool actually returned — findings must be
    traceable back to one of these, never invented.
    """

    title: str = Field(description="Title of the source page, as returned by search.")
    url: str = Field(description="Exact URL of the source, as returned by search.")
    domain: str = Field(default="", description="Domain extracted from the URL, for dedupe/credibility checks.")
    content: str = Field(description="Snippet/content extracted from the source.")
    published_date: Optional[str] = Field(
        default=None, description="Publication date if the search provider supplied one."
    )
    credibility_hint: str = Field(
        default="unknown",
        description="Soft, general-purpose heuristic ('high'/'unknown') based on domain type "
                    "(government/intergovernmental/academic/etc). Never used to silently drop a source.",
    )


class Finding(BaseModel):
    """A single extracted, source-backed claim."""

    claim: str = Field(description="A specific factual claim relevant to the research task.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well-supported the claim is by the cited evidence."
    )
    evidence: str = Field(description="The supporting text the claim is based on.")
    relevance: str = Field(description="Why this finding matters to the research task.")
    source: str = Field(description="Name/title of the source, taken from the search result.")
    url: str = Field(description="Exact source URL, copied verbatim from the search result.")


class ResearchResult(BaseModel):
    """Output of researching one task (or the merged output of an entire round)."""

    task_id: str = Field(description="Identifier of the task this result answers.")
    findings: List[Finding] = Field(default_factory=list)
    summary: str = Field(default="", description="Concise summary based only on the findings above.")
    sources: List[SourceEvidence] = Field(
        default_factory=list,
        description="Every raw source collected for this task, for traceability — "
                    "independent of whether the LLM successfully extracted a Finding from it.",
    )
    searches_used: int = Field(
        default=0, description="Number of actual web searches performed for this task, "
                                "used by the Orchestrator to enforce MAX_TOTAL_SEARCHES."
    )


# ============================================================
# Review (produced by the Reviewer)
# ============================================================

class ReviewResult(BaseModel):
    approved: bool = Field(description="Whether the research is good enough to be treated as final.")
    score: int = Field(description="Overall research quality score, 0-100.")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(
        default_factory=list,
        description="Specific, actionable gaps to research next. These are converted "
                    "directly into new ResearchTasks for the next round.",
    )
    summary: str = Field(default="", description="Concise explanation of the review decision.")


# ============================================================
# Final Report (produced by the Report Generator)
# ============================================================

class FinalReport(BaseModel):
    """
    The final deliverable. `narrative_markdown` is the only LLM-authored
    part — it adapts structure (tables for comparisons, timelines for
    historical questions, etc.) to the actual question. Everything else
    (sources, findings, methodology, limitations) is built deterministically
    from data already validated earlier in the pipeline, so the report can
    never contain a fact that isn't traceable to a real Finding/Source.
    """

    research_question: str = Field(description="The original research question.")
    approved: bool = Field(description="Whether the underlying research was approved by the Reviewer.")
    score: int = Field(description="The Reviewer's final quality score, 0-100.")

    narrative_markdown: str = Field(
        description="LLM-authored markdown covering Executive Summary, Key Findings, "
                    "Detailed Findings/Analysis, and Conclusion — structure adapted to the topic."
    )

    methodology: str = Field(description="Deterministic description of how the research was conducted.")
    limitations: List[str] = Field(default_factory=list, description="Known gaps or caveats.")

    findings: List[Finding] = Field(default_factory=list)
    sources: List[SourceEvidence] = Field(default_factory=list)

    generated_at: str = Field(description="ISO-8601 timestamp of report generation.")
