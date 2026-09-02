import json
import os
import tempfile

import agents.report_generator as report_module
from models.schemas import Finding, ResearchPlan, ResearchResult, ResearchTask, ReviewResult, SourceEvidence


def _plan():
    return ResearchPlan(
        research_question="Compare RAG and fine-tuning for enterprise AI.",
        objectives=["Compare cost", "Compare accuracy"],
        tasks=[ResearchTask(task_id="T1", name="Cost", description="Compare cost.")],
    )


def _research():
    return ResearchResult(
        task_id="aggregated",
        findings=[
            Finding(claim="RAG is cheaper upfront.", confidence="high", evidence="e", relevance="r", source="Vendor Docs", url="https://example.com/a")
        ],
        sources=[SourceEvidence(title="Vendor Docs", url="https://example.com/a", domain="example.com", content="c")],
        searches_used=2,
    )


def _review(approved=True):
    return ReviewResult(approved=approved, score=85 if approved else 40, summary="Solid.", weaknesses=[] if approved else ["missing accuracy data"])


def test_build_final_report_uses_llm_narrative_on_success(monkeypatch):
    class _FakeLLM:
        def invoke(self, messages):
            class _Resp:
                content = "## Executive Summary\n\nRAG is cheaper.\n"
            return _Resp()

    monkeypatch.setattr(report_module, "_get_narrative_llm", lambda: _FakeLLM())

    report = report_module.build_final_report(_plan(), _research(), _review())

    assert report.approved is True
    assert report.score == 85
    assert "RAG is cheaper" in report.narrative_markdown
    assert len(report.findings) == 1
    assert len(report.sources) == 1
    assert report.generated_at  # non-empty ISO timestamp


def test_build_final_report_falls_back_deterministically_when_llm_fails(monkeypatch):
    class _AlwaysFailsLLM:
        def invoke(self, messages):
            raise Exception("401 invalid api key")

    monkeypatch.setattr(report_module, "_get_narrative_llm", lambda: _AlwaysFailsLLM())

    report = report_module.build_final_report(_plan(), _research(), _review(approved=False))

    # Fallback narrative must still be built from the real findings, not invented.
    assert "RAG is cheaper upfront." in report.narrative_markdown
    assert report.approved is False


def test_save_report_writes_all_three_formats(monkeypatch):
    class _FakeLLM:
        def invoke(self, messages):
            class _Resp:
                content = "## Executive Summary\n\nSummary text.\n"
            return _Resp()

    monkeypatch.setattr(report_module, "_get_narrative_llm", lambda: _FakeLLM())

    report = report_module.build_final_report(_plan(), _research(), _review())

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = report_module.save_report(report, output_dir=tmpdir)

        assert os.path.isfile(paths["markdown"])
        assert os.path.isfile(paths["json"])
        assert os.path.isfile(paths["html"])

        with open(paths["markdown"], encoding="utf-8") as f:
            md_content = f.read()
        assert "# Research Report" in md_content
        assert "example.com/a" in md_content  # source traceable in output

        with open(paths["json"], encoding="utf-8") as f:
            data = json.load(f)
        assert data["research_question"] == "Compare RAG and fine-tuning for enterprise AI."
        assert data["findings"][0]["url"] == "https://example.com/a"

        with open(paths["html"], encoding="utf-8") as f:
            html_content = f.read()
        assert "<html" in html_content.lower()
