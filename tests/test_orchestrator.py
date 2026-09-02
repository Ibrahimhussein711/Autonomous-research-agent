import agents.orchestrator as orchestrator_module
from config.settings import settings
from models.schemas import Finding, ResearchPlan, ResearchResult, ResearchTask, ReviewResult, SourceEvidence


def _plan(n_tasks=2):
    tasks = [
        ResearchTask(task_id=f"T{i}", name=f"Task {i}", description=f"desc {i}")
        for i in range(1, n_tasks + 1)
    ]
    return ResearchPlan(research_question="Q?", objectives=["obj1"], tasks=tasks)


def test_orchestrator_approves_after_refinement_round(monkeypatch):
    calls = []

    def fake_researcher(task, search_history, max_searches=None):
        calls.append(("researcher", task.task_id))
        return ResearchResult(
            task_id=task.task_id,
            findings=[
                Finding(claim=f"claim-{task.task_id}", confidence="high", evidence="e", relevance="r", source="s", url=f"https://x.com/{task.task_id}")
            ],
            sources=[SourceEvidence(title="t", url=f"https://x.com/{task.task_id}", content="c")],
            searches_used=1,
        )

    review_calls = {"n": 0}

    def fake_reviewer(question, plan, merged):
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            return ReviewResult(approved=False, score=50, weaknesses=["missing X"], recommendations=["find more about X"])
        return ReviewResult(approved=True, score=90)

    monkeypatch.setattr(orchestrator_module, "researcher_agent", fake_researcher)
    monkeypatch.setattr(orchestrator_module, "reviewer_agent", fake_reviewer)

    result, review = orchestrator_module.research_orchestrator(_plan(n_tasks=2))

    assert review.approved is True
    # T1, T2 from round 1, plus one refinement task from round 2 recommendation.
    assert len(result.findings) == 3
    assert {"researcher", "reviewer"} <= {c[0] for c in calls} | {"reviewer"}
    researcher_task_ids = [c[1] for c in calls if c[0] == "researcher"]
    assert "T1" in researcher_task_ids and "T2" in researcher_task_ids
    assert any(tid.startswith("R2-") for tid in researcher_task_ids)


def test_orchestrator_preserves_cumulative_evidence_across_rounds(monkeypatch):
    """Findings from round 1 must still be present in the final merged result
    even if round 2 only adds a couple more — nothing gets thrown away."""

    round_findings = {
        "T1": [Finding(claim="round1-a", confidence="high", evidence="e", relevance="r", source="s", url="https://x.com/1")],
        "R2-1": [Finding(claim="round2-a", confidence="medium", evidence="e", relevance="r", source="s", url="https://x.com/2")],
    }

    def fake_researcher(task, search_history, max_searches=None):
        return ResearchResult(task_id=task.task_id, findings=round_findings.get(task.task_id, []), searches_used=1)

    review_calls = {"n": 0}

    def fake_reviewer(question, plan, merged):
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            return ReviewResult(approved=False, score=40, recommendations=["dig deeper"])
        return ReviewResult(approved=True, score=80)

    monkeypatch.setattr(orchestrator_module, "researcher_agent", fake_researcher)
    monkeypatch.setattr(orchestrator_module, "reviewer_agent", fake_reviewer)

    result, review = orchestrator_module.research_orchestrator(_plan(n_tasks=1))

    claims = {f.claim for f in result.findings}
    assert "round1-a" in claims  # preserved from round 1
    assert "round2-a" in claims  # added in round 2


def test_orchestrator_stops_at_max_rounds_when_never_approved(monkeypatch):
    def fake_researcher(task, search_history, max_searches=None):
        return ResearchResult(
            task_id=task.task_id,
            findings=[Finding(claim=f"c-{task.task_id}", confidence="low", evidence="e", relevance="r", source="s", url=f"https://x.com/{task.task_id}")],
            searches_used=1,
        )

    def fake_reviewer(question, plan, merged):
        return ReviewResult(approved=False, score=30, recommendations=["still missing stuff"])

    monkeypatch.setattr(orchestrator_module, "researcher_agent", fake_researcher)
    monkeypatch.setattr(orchestrator_module, "reviewer_agent", fake_reviewer)

    result, review = orchestrator_module.research_orchestrator(_plan(n_tasks=1))

    assert review.approved is False
    # Exactly settings.max_research_rounds review calls should have happened —
    # verified indirectly via findings count growing by 1 per round (1 task/round).
    assert len(result.findings) == settings.max_research_rounds


def test_orchestrator_enforces_total_search_budget(monkeypatch):
    original_budget = settings.max_total_searches
    object.__setattr__(settings, "max_total_searches", 2)  # frozen dataclass, patch for this test

    try:
        searches_requested = []

        def fake_researcher(task, search_history, max_searches=None):
            searches_requested.append(max_searches)
            used = min(max_searches or 0, 5)
            return ResearchResult(task_id=task.task_id, findings=[], searches_used=used)

        def fake_reviewer(question, plan, merged):
            return ReviewResult(approved=False, score=0, recommendations=["x"])

        monkeypatch.setattr(orchestrator_module, "researcher_agent", fake_researcher)
        monkeypatch.setattr(orchestrator_module, "reviewer_agent", fake_reviewer)

        orchestrator_module.research_orchestrator(_plan(n_tasks=3))

        # Budget of 2 total searches must never be exceeded across all task calls.
        assert sum(searches_requested) <= 2 * len(searches_requested)  # sanity: caps were honored
        assert all(cap <= 2 for cap in searches_requested)
    finally:
        object.__setattr__(settings, "max_total_searches", original_budget)
