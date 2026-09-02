import json

from pydantic import BaseModel

from utils.structured_output import find_first_json_object, invoke_structured


class _DummySchema(BaseModel):
    name: str
    score: int


def test_find_first_json_object_extracts_balanced_block():
    text = 'some preamble {"a": 1, "b": {"c": 2}} trailing junk'
    block = find_first_json_object(text)
    assert json.loads(block) == {"a": 1, "b": {"c": 2}}


def test_find_first_json_object_returns_none_when_absent():
    assert find_first_json_object("no json here") is None


class _NativeStructuredLLM:
    """Simulates a provider whose .with_structured_output().invoke() works fine."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema_cls):
        return self

    def invoke(self, messages):
        return self._result


def test_invoke_structured_uses_native_path_when_it_works():
    expected = _DummySchema(name="x", score=5)
    llm = _NativeStructuredLLM(expected)

    result = invoke_structured(llm, [("system", "s"), ("human", "h")], _DummySchema, "{...}")

    assert result == expected


class _BrokenNativeThenPlainLLM:
    """Simulates native structured output failing, then a plain .invoke() succeeding."""

    def __init__(self, plain_response_text):
        self._plain_response_text = plain_response_text
        self.plain_invoke_calls = 0

    def with_structured_output(self, schema_cls):
        raise RuntimeError("this provider doesn't support that method")

    def invoke(self, messages):
        self.plain_invoke_calls += 1

        class _Resp:
            content = self._plain_response_text

        return _Resp()


def test_invoke_structured_falls_back_to_manual_json_parsing():
    llm = _BrokenNativeThenPlainLLM('Sure, here it is: {"name": "y", "score": 9} done.')

    result = invoke_structured(llm, [("system", "s"), ("human", "h")], _DummySchema, "{...}")

    assert isinstance(result, _DummySchema)
    assert result.name == "y"
    assert result.score == 9
    assert llm.plain_invoke_calls == 1


class _AlwaysBrokenLLM:
    def with_structured_output(self, schema_cls):
        raise RuntimeError("unsupported")

    def invoke(self, messages):
        class _Resp:
            content = "no json anywhere in this response"

        return _Resp()


def test_invoke_structured_raises_when_both_paths_fail():
    llm = _AlwaysBrokenLLM()
    try:
        invoke_structured(llm, [("system", "s")], _DummySchema, "{...}")
        assert False, "expected an exception"
    except ValueError:
        pass
