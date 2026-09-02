"""
CLI entry point.

Deliberately thin: read a question, call the Planner, the Orchestrator,
then the Report Generator, and print/save the result. No agent logic
lives here.
"""

import sys

from agents.orchestrator import research_orchestrator
from agents.planner import planner_agent
from agents.report_generator import build_final_report, save_report
from config.settings import settings
from utils import logging as log
from utils.ollama_check import check_ollama_available


def main() -> None:
    log.section("AUTONOMOUS RESEARCH AGENT")

    problems = settings.validate()
    if problems:
        for p in problems:
            log.error(p)
        sys.exit(1)

    log.info(f"Checking Ollama at {settings.ollama_base_url} for model '{settings.ollama_model}'...")
    ollama_ok, ollama_message = check_ollama_available()
    if ollama_ok:
        log.success(ollama_message)
    else:
        log.error(ollama_message)
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        print("\nEnter your research question:")
        question = input("> ").strip()

    if not question:
        log.error("No research question provided.")
        sys.exit(1)

    print(f"\nQUESTION:\n{question}")

    # ========================================================
    # STEP 1 — PLANNING
    # ========================================================
    log.section("STEP 1 — PLANNING")

    plan = planner_agent(question)

    log.subsection("📋 RESEARCH PLAN")
    print(plan.model_dump_json(indent=2))

    log.subsection("📋 TASKS")
    for t in plan.tasks:
        print(f"{t.task_id} - {t.name}: {t.description}")

    print(f"\nPlanner generated {len(plan.tasks)} task(s).")

    # ========================================================
    # STEP 2 — AUTONOMOUS RESEARCH (Research -> Review -> Refine)
    # ========================================================
    log.section("STEP 2 — RESEARCH")

    research_result, review_result = research_orchestrator(plan)

    # ========================================================
    # STEP 3 — REVIEW (final)
    # ========================================================
    log.section("STEP 3 — REVIEW")
    print(f"Score: {review_result.score}/100")
    print(f"Approved: {'YES' if review_result.approved else 'NO'}")

    # ========================================================
    # STEP 4 — REPORT GENERATION
    # ========================================================
    log.section("STEP 4 — REPORT GENERATION")
    log.info("Generating final report...")

    report = build_final_report(plan, research_result, review_result)
    paths = save_report(report)

    # ========================================================
    # FINAL RESULT
    # ========================================================
    log.section("FINAL RESULT")

    print(f"Status: {'APPROVED' if review_result.approved else 'NOT APPROVED (max rounds/budget reached)'}")
    print(f"Score: {review_result.score}/100")
    print(f"Findings: {len(research_result.findings)}")
    print(f"Sources: {len(research_result.sources)}")
    print(f"Searches used: {research_result.searches_used}/{settings.max_total_searches}")
    print("\nReport saved to:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
