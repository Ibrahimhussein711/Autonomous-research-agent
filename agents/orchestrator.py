"""
Research Orchestrator.

Drives the full Research -> Review -> Refine loop for an entire
ResearchPlan (NOT a single opaque string). This is the piece that ties
the Planner's dynamic tasks to the Researcher and Reviewer:

    ResearchPlan.tasks --(round 1)--> Researcher (per task) --> merge
        --> Reviewer --> approved? done : recommendations
        --(round 2)--> new tasks built FROM the recommendations --> ...

The Orchestrator itself never searches the web and never judges quality —
it only sequences the other three agents and keeps the running state
that spans across rounds and tasks: merged findings/sources (cumulative
— nothing from an earlier round is thrown away), a search-history set to
avoid duplicate queries, and a global search budget (MAX_TOTAL_SEARCHES)
enforced on top of the per-task limit.
"""

from typing import List, Set, Tuple

from config.settings import settings
from agents.researcher import researcher_agent
from agents.reviewer import reviewer_agent
from models.schemas import Finding, ResearchPlan, ResearchResult, ResearchTask, ReviewResult, SourceEvidence
from utils import logging as log

AGGREGATE_TASK_ID = "aggregated"


def _dedupe_findings(findings: List[Finding]) -> List[Finding]:
    seen: Set[Tuple[str, str]] = set()
    unique: List[Finding] = []
    for f in findings:
        key = (f.url.strip().lower(), f.claim.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _dedupe_sources(sources: List[SourceEvidence]) -> List[SourceEvidence]:
    seen: Set[str] = set()
    unique: List[SourceEvidence] = []
    for s in sources:
        if s.url and s.url not in seen:
            seen.add(s.url)
            unique.append(s)
    return unique


def _tasks_from_recommendations(recommendations: List[str], round_number: int) -> List[ResearchTask]:
    """
    Turn the Reviewer's recommendations into new, standalone research
    tasks for the next round. This is the "refinement" step from the
    spec — deliberately dynamic, built entirely from what the Reviewer
    said was missing rather than any fixed topic template.
    """
    tasks = []
    for i, recommendation in enumerate(recommendations, start=1):
        tasks.append(
            ResearchTask(
                task_id=f"R{round_number}-{i}",
                name=f"Refinement {i}",
                description=recommendation,
                recommended_source_types=[
                    "authoritative primary sources relevant to this specific gap",
                    "recent, reputable reporting or official documentation",
                ],
            )
        )
    return tasks


def research_orchestrator(plan: ResearchPlan) -> Tuple[ResearchResult, ReviewResult]:
    log.section("AUTONOMOUS RESEARCH ORCHESTRATOR")

    search_history: Set[str] = set()
    all_findings: List[Finding] = []
    all_sources: List[SourceEvidence] = []
    total_searches_used = 0

    current_tasks: List[ResearchTask] = list(plan.tasks)
    final_review = ReviewResult(approved=False, score=0, summary="No review has run yet.")

    for round_number in range(1, settings.max_research_rounds + 1):
        log.section(f"RESEARCH ROUND {round_number}")

        if not current_tasks:
            log.warn("No tasks to research this round — stopping early.")
            break

        for task_index, task in enumerate(current_tasks, start=1):
            remaining_budget = settings.max_total_searches - total_searches_used
            if remaining_budget <= 0:
                log.warn(
                    f"Global search budget ({settings.max_total_searches}) exhausted — "
                    f"skipping remaining tasks this round."
                )
                break

            log.info(f"Task {task_index}/{len(current_tasks)} this round (global searches used: {total_searches_used}/{settings.max_total_searches})")

            per_task_cap = min(settings.max_searches_per_task, remaining_budget)
            result = researcher_agent(task, search_history, max_searches=per_task_cap)

            total_searches_used += result.searches_used
            all_findings.extend(result.findings)
            all_sources.extend(result.sources)

        # Cumulative — nothing from earlier rounds/tasks is ever dropped here.
        all_findings = _dedupe_findings(all_findings)
        all_sources = _dedupe_sources(all_sources)

        merged_result = ResearchResult(
            task_id=AGGREGATE_TASK_ID,
            findings=all_findings,
            summary=(
                f"{len(all_findings)} finding(s) collected across {round_number} round(s) "
                f"for: {plan.research_question}"
            ),
            sources=all_sources,
            searches_used=total_searches_used,
        )

        log.subsection("📚 MERGED RESEARCH RESULT")
        print(
            f"Findings so far: {len(all_findings)}  |  Sources so far: {len(all_sources)}  |  "
            f"Searches used: {total_searches_used}/{settings.max_total_searches}"
        )

        if not all_findings:
            log.warn("No findings produced yet.")
            if round_number >= settings.max_research_rounds or total_searches_used >= settings.max_total_searches:
                final_review = ReviewResult(
                    approved=False,
                    score=0,
                    weaknesses=["No verified research findings were produced in any round."],
                    recommendations=["Retry after checking API keys, connectivity, and search budget."],
                    summary="Research failed before producing usable evidence.",
                )
                return merged_result, final_review
            continue

        review_result = reviewer_agent(plan.research_question, plan, merged_result)
        final_review = review_result

        log.subsection("🔍 REVIEW RESULT")
        print(review_result.model_dump_json(indent=2))

        if review_result.approved:
            log.section("FINAL DECISION")
            log.success(f"Research APPROVED — score {review_result.score}/100")
            return merged_result, final_review

        if round_number >= settings.max_research_rounds:
            log.section("FINAL DECISION")
            log.warn(f"Maximum research rounds reached. Final score: {review_result.score}/100")
            return merged_result, final_review

        if total_searches_used >= settings.max_total_searches:
            log.section("FINAL DECISION")
            log.warn(f"Global search budget exhausted. Final score: {review_result.score}/100")
            return merged_result, final_review

        log.section("REFINING RESEARCH")
        for rec in review_result.recommendations:
            print(f"- {rec}")

        current_tasks = _tasks_from_recommendations(review_result.recommendations, round_number + 1)

        if not current_tasks:
            log.warn("Reviewer gave no actionable recommendations — stopping.")
            return merged_result, final_review

    merged_result = ResearchResult(
        task_id=AGGREGATE_TASK_ID,
        findings=all_findings,
        summary=f"{len(all_findings)} finding(s) collected for: {plan.research_question}",
        sources=all_sources,
        searches_used=total_searches_used,
    )
    return merged_result, final_review
