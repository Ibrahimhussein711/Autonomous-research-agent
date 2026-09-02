import pytest
from pydantic import ValidationError

from models.schemas import (
    Finding,
    FinalReport,
    ResearchPlan,
    ResearchResult,
    ResearchTask,
    ReviewResult,
    SourceEvidence,
)


def test_research_task_requires_fields():
    task = ResearchTask(task_id="T1", name="Costs", description="Research costs.")
    assert task.recommended_source_types == []


def test_research_plan_holds_tasks():
    plan = ResearchPlan(
        research_question="Q?",
        objectives=["obj1"],
        tasks=[ResearchTask(task_id="T1", name="n", description="d")],
    )
    assert len(plan.tasks) == 1
    assert plan.research_question == "Q?"


def test_finding_confidence_must_be_valid_literal():
    with pytest.raises(ValidationError):
        Finding(
            claim="c", confidence="super-high", evidence="e",
            relevance="r", source="s", url="https://x.com",
        )


def test_finding_valid_confidence_values():
    for level in ("high", "medium", "low"):
        f = Finding(claim="c", confidence=level, evidence="e", relevance="r", source="s", url="https://x.com")
        assert f.confidence == level


def test_research_result_defaults():
    result = ResearchResult(task_id="T1")
    assert result.findings == []
    assert result.sources == []
    assert result.searches_used == 0


def test_review_result_scoring_bounds_are_just_ints():
    review = ReviewResult(approved=True, score=90)
    assert review.approved is True
    assert review.score == 90


def test_source_evidence_defaults():
    s = SourceEvidence(title="t", url="https://x.com/a", content="c")
    assert s.credibility_hint == "unknown"
    assert s.domain == ""


def test_final_report_requires_generated_at():
    with pytest.raises(ValidationError):
        FinalReport(
            research_question="Q?",
            approved=True,
            score=90,
            narrative_markdown="body",
            methodology="m",
        )


def test_final_report_valid():
    report = FinalReport(
        research_question="Q?",
        approved=True,
        score=90,
        narrative_markdown="body",
        methodology="m",
        generated_at="2026-01-01T00:00:00Z",
    )
    assert report.score == 90
    assert report.findings == []
