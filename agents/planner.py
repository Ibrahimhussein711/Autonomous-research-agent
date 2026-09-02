"""
Planner Agent.

Turns an arbitrary research question into a ResearchPlan: a set of
objectives and complementary ResearchTasks. The Planner never searches
the web and never answers the question — it only decides *what* needs
to be researched and *what kind* of sources are likely to help.

IMPORTANT: nothing in this file may reference a specific topic (energy,
AI, finance, etc). The plan must be derived from whatever question is
passed in at runtime, including in the failure-fallback path.
"""

from langchain_ollama import ChatOllama

from config.settings import settings
from models.schemas import ResearchPlan, ResearchTask
from utils import logging as log
from utils.retry import call_with_retry
from utils.structured_output import invoke_structured

_raw_llm = None


def _get_raw_llm():
    """Lazily construct the LLM so importing this module never requires Ollama to be running."""
    global _raw_llm
    if _raw_llm is None:
        _raw_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=settings.planner_max_output_tokens,
        )
    return _raw_llm


SYSTEM_PROMPT = """
You are a research planning agent.

Your job is to convert the user's research question into a concise,
actionable research plan. You do NOT answer the question and you do NOT
perform any research yourself — you only decide what needs to be
researched.

The plan must contain:
1. research_question — copy the user's question.
2. objectives — the main things the research must establish.
3. tasks — 3 to 6 complementary research tasks that together cover the
   objectives.

Each task must contain:
- task_id (e.g. "T1", "T2", ...)
- name — a short label
- description — a specific, actionable research instruction
- recommended_source_types — the kinds of sources likely to have good
  evidence for THIS task specifically (e.g. "peer-reviewed studies",
  "government statistics agencies", "official vendor documentation",
  "recent technical journalism"). Choose these based on the actual
  subject matter of the question, not a fixed list.

CRITICAL RULES:
- The tasks must be derived entirely from the specific question you are
  given. Do not reuse a generic template like "benefits / challenges /
  sources / applications" unless the question is genuinely about weighing
  pros and cons of something — most questions are not.
- A comparison question needs tasks that compare each dimension
  (e.g. cost, performance, architecture) between the named things.
- A causal/historical question ("what caused X") needs tasks that dig
  into distinct causes, mechanisms, and consequences.
- A "recent developments in X" question needs tasks structured around
  distinct subtopics or timeframes, with an explicit emphasis on recency.
- Two different research questions should almost always produce
  different task sets. If you notice yourself writing the same four
  tasks regardless of the question, stop and reconsider.
- Keep the plan focused — do not create unnecessary or redundant tasks.
"""

_PLAN_JSON_SHAPE = """{
  "research_question": "...",
  "objectives": ["...", "..."],
  "tasks": [
    {
      "task_id": "T1",
      "name": "...",
      "description": "...",
      "recommended_source_types": ["...", "..."]
    }
  ]
}"""


def _build_fallback_plan(question: str) -> ResearchPlan:
    """
    Used only if the LLM call fails after all retries.

    This must NOT hardcode any topic. It produces a single broad task
    built directly from the user's own question text, so the rest of
    the pipeline (Researcher/Reviewer) still has something reasonable
    to work with instead of crashing.
    """
    return ResearchPlan(
        research_question=question,
        objectives=[
            f"Directly and thoroughly answer the research question: {question}"
        ],
        tasks=[
            ResearchTask(
                task_id="T1",
                name="Primary research",
                description=(
                    f"Research and gather evidence that directly answers: {question}. "
                    "Cover the main aspects of the question and collect concrete, "
                    "recent, and verifiable evidence."
                ),
                recommended_source_types=[
                    "authoritative primary sources relevant to the question",
                    "recent, reputable reporting or documentation",
                ],
            )
        ],
    )


def planner_agent(question: str) -> ResearchPlan:
    log.section("PLANNER AGENT")

    def _invoke():
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", question),
        ]
        return invoke_structured(_get_raw_llm(), messages, ResearchPlan, _PLAN_JSON_SHAPE)

    outcome = call_with_retry(
        _invoke,
        max_retries=settings.planner_max_retries,
        base_delay=settings.retry_base_delay_seconds,
        max_delay=settings.retry_max_delay_seconds,
        on_retry=log.retrying,
    )

    if outcome.success:
        log.success(f"Research plan created with {len(outcome.value.tasks)} task(s).")
        return outcome.value

    log.error(f"Planner failed after {outcome.attempts} attempt(s) [{outcome.kind}]: {outcome.error}")
    log.warn("Falling back to a single broad research task derived from the question.")

    return _build_fallback_plan(question)
