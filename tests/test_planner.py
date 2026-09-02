from models.schemas import ResearchPlan, ResearchTask

import agents.planner as planner_module


def test_planner_returns_llm_plan_on_success(monkeypatch):
    expected = ResearchPlan(
        research_question="Compare RAG and fine-tuning.",
        objectives=["Understand tradeoffs"],
        tasks=[
            ResearchTask(task_id="T1", name="Cost", description="Compare cost."),
            ResearchTask(task_id="T2", name="Accuracy", description="Compare accuracy."),
        ],
    )
    calls = {"n": 0}

    def fake_invoke_structured(raw_llm, messages, schema_cls, json_shape_hint):
        calls["n"] += 1
        assert schema_cls is ResearchPlan
        return expected

    monkeypatch.setattr(planner_module, "invoke_structured", fake_invoke_structured)
    monkeypatch.setattr(planner_module, "_get_raw_llm", lambda: object())

    plan = planner_module.planner_agent("Compare RAG and fine-tuning.")

    assert plan.research_question == "Compare RAG and fine-tuning."
    assert len(plan.tasks) == 2
    assert calls["n"] == 1


def test_planner_falls_back_on_persistent_failure(monkeypatch):
    def always_fails(raw_llm, messages, schema_cls, json_shape_hint):
        raise Exception("connection refused")

    monkeypatch.setattr(planner_module, "invoke_structured", always_fails)
    monkeypatch.setattr(planner_module, "_get_raw_llm", lambda: object())

    question = "What caused the 2008 financial crisis?"
    plan = planner_module.planner_agent(question)

    # Fallback must be derived from the question, not any hardcoded topic.
    assert plan.research_question == question
    assert len(plan.tasks) == 1
    assert question in plan.tasks[0].description
    assert "renewable" not in plan.tasks[0].description.lower()
    assert "benefit" not in plan.tasks[0].description.lower()


def test_planner_different_questions_are_not_forced_into_same_fallback_task(monkeypatch):
    def always_fails(raw_llm, messages, schema_cls, json_shape_hint):
        raise Exception("connection refused")

    monkeypatch.setattr(planner_module, "invoke_structured", always_fails)
    monkeypatch.setattr(planner_module, "_get_raw_llm", lambda: object())

    plan_a = planner_module.planner_agent("Question A about robotics")
    plan_b = planner_module.planner_agent("Question B about finance")

    assert plan_a.tasks[0].description != plan_b.tasks[0].description
