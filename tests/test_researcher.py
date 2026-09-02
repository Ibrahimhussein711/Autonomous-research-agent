import json

import agents.researcher as researcher_module
from models.schemas import ResearchTask, SourceEvidence


class _FakeAIMessage:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


class _SequenceLLM:
    """Returns each response in order, one per .invoke() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def _task():
    return ResearchTask(
        task_id="T1",
        name="Cost comparison",
        description="Compare the cost of RAG vs fine-tuning.",
        recommended_source_types=["technical blogs", "vendor docs"],
    )


def test_researcher_performs_search_and_extracts_findings(monkeypatch):
    tool_call_msg = _FakeAIMessage(
        tool_calls=[{"name": "search_web", "args": {"query": "RAG vs fine-tuning cost"}, "id": "call_1"}]
    )
    final_msg = _FakeAIMessage(tool_calls=[])

    fake_chat_llm = _SequenceLLM([tool_call_msg, final_msg])
    monkeypatch.setattr(researcher_module, "_get_llm_with_tools", lambda: fake_chat_llm)

    fake_evidence = [
        SourceEvidence(title="Cost Guide", url="https://example.com/cost", content="RAG is cheaper for small corpora.")
    ]
    monkeypatch.setattr(researcher_module, "search_web_raw", lambda query: fake_evidence)

    extraction_payload = {
        "task_id": "T1",
        "findings": [
            {
                "claim": "RAG is cheaper for small corpora.",
                "confidence": "high",
                "evidence": "RAG is cheaper for small corpora.",
                "relevance": "Directly answers the cost comparison task.",
                "source": "Cost Guide",
                "url": "https://example.com/cost",
            }
        ],
        "summary": "One cost-related finding extracted.",
    }
    fake_extractor = _SequenceLLM([_FakeAIMessage(content=json.dumps(extraction_payload))])
    monkeypatch.setattr(researcher_module, "_get_extractor_llm", lambda: fake_extractor)

    result = researcher_module.researcher_agent(_task(), search_history=set())

    assert result.task_id == "T1"
    assert len(result.findings) == 1
    assert result.findings[0].url == "https://example.com/cost"
    assert result.searches_used == 1
    assert len(result.sources) == 1


def test_researcher_respects_max_searches_cap(monkeypatch):
    # LLM keeps requesting NEW searches forever; researcher must stop at the cap
    # (each query is unique so this exercises the cap, not the dedup feature).
    infinite_tool_calls = [
        _FakeAIMessage(tool_calls=[{"name": "search_web", "args": {"query": f"q{i}"}, "id": f"call_{i}"}])
        for i in range(10)
    ]
    fake_chat_llm = _SequenceLLM(infinite_tool_calls)
    monkeypatch.setattr(researcher_module, "_get_llm_with_tools", lambda: fake_chat_llm)

    call_count = {"n": 0}

    def fake_search(query):
        call_count["n"] += 1
        return [SourceEvidence(title=f"T{call_count['n']}", url=f"https://x.com/{call_count['n']}", content="c")]

    monkeypatch.setattr(researcher_module, "search_web_raw", fake_search)
    monkeypatch.setattr(
        researcher_module,
        "_get_extractor_llm",
        lambda: _SequenceLLM([_FakeAIMessage(content='{"task_id":"T1","findings":[],"summary":"s"}')] * 5),
    )

    result = researcher_module.researcher_agent(_task(), search_history=set(), max_searches=2)

    assert result.searches_used == 2
    assert call_count["n"] == 2


def test_researcher_avoids_duplicate_queries(monkeypatch):
    tool_call_msg = _FakeAIMessage(
        tool_calls=[{"name": "search_web", "args": {"query": "already searched"}, "id": "call_1"}]
    )
    final_msg = _FakeAIMessage(tool_calls=[])
    fake_chat_llm = _SequenceLLM([tool_call_msg, final_msg])
    monkeypatch.setattr(researcher_module, "_get_llm_with_tools", lambda: fake_chat_llm)

    search_called = {"n": 0}

    def fake_search(query):
        search_called["n"] += 1
        return [SourceEvidence(title="t", url="https://x.com", content="c")]

    monkeypatch.setattr(researcher_module, "search_web_raw", fake_search)

    history = {"already searched"}
    result = researcher_module.researcher_agent(_task(), search_history=history)

    assert search_called["n"] == 0
    assert result.searches_used == 0


def test_researcher_salvages_findings_when_extraction_fails(monkeypatch):
    tool_call_msg = _FakeAIMessage(
        tool_calls=[{"name": "search_web", "args": {"query": "q"}, "id": "call_1"}]
    )
    final_msg = _FakeAIMessage(tool_calls=[])
    fake_chat_llm = _SequenceLLM([tool_call_msg, final_msg])
    monkeypatch.setattr(researcher_module, "_get_llm_with_tools", lambda: fake_chat_llm)

    fake_evidence = [SourceEvidence(title="Raw Source", url="https://example.com/raw", content="some raw content")]
    monkeypatch.setattr(researcher_module, "search_web_raw", lambda query: fake_evidence)

    # Extractor always returns garbage that can't be parsed as JSON at all.
    bad_extractor = _SequenceLLM([_FakeAIMessage(content="not json at all") for _ in range(5)])
    monkeypatch.setattr(researcher_module, "_get_extractor_llm", lambda: bad_extractor)

    result = researcher_module.researcher_agent(_task(), search_history=set())

    # Evidence must be preserved even though structured extraction totally failed.
    assert len(result.findings) == 1
    assert result.findings[0].url == "https://example.com/raw"
    assert result.findings[0].confidence == "low"
