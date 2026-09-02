import agents.reviewer as reviewer_module
from models.schemas import Finding, ResearchPlan, ResearchResult, ResearchTask, ReviewResult


def _plan():
    return ResearchPlan(
        research_question="Q?",
        objectives=["obj1"],
        tasks=[ResearchTask(task_id="T1", name="n", description="d")],
    )


def test_reviewer_returns_zero_score_for_empty_findings():
    review = reviewer_module.reviewer_agent("Q?", _plan(), ResearchResult(task_id="T1", findings=[]))
    assert review.approved is False
    assert review.score == 0
    # Must not call the LLM at all for empty findings.


def test_reviewer_returns_llm_result_on_success(monkeypatch):
    expected = ReviewResult(approved=True, score=88, summary="Solid research.")

    def fake_invoke_structured(raw_llm, messages, schema_cls, json_shape_hint):
        assert schema_cls is ReviewResult
        return expected

    monkeypatch.setattr(reviewer_module, "invoke_structured", fake_invoke_structured)
    monkeypatch.setattr(reviewer_module, "_get_raw_llm", lambda: object())

    research = ResearchResult(
        task_id="T1",
        findings=[Finding(claim="c", confidence="high", evidence="e", relevance="r", source="s", url="https://x.com")],
    )
    review = reviewer_module.reviewer_agent("Q?", _plan(), research)

    assert review.approved is True
    assert review.score == 88


def test_reviewer_fails_gracefully_on_persistent_llm_failure(monkeypatch):
    def always_fails(raw_llm, messages, schema_cls, json_shape_hint):
        raise Exception("connection refused")

    monkeypatch.setattr(reviewer_module, "invoke_structured", always_fails)
    monkeypatch.setattr(reviewer_module, "_get_raw_llm", lambda: object())

    research = ResearchResult(
        task_id="T1",
        findings=[Finding(claim="c", confidence="high", evidence="e", relevance="r", source="s", url="https://x.com")],
    )
    review = reviewer_module.reviewer_agent("Q?", _plan(), research)

    assert review.approved is False
    assert review.score == 0
    assert any("Reviewer execution failed" in w for w in review.weaknesses)
