import time

from utils.retry import (
    FailureKind,
    call_with_retry,
    classify_error,
    extract_retry_after_seconds,
    is_retryable,
)


def test_classify_rate_limit():
    assert classify_error(Exception("Error code: 429 - rate limit reached")) == FailureKind.RATE_LIMIT


def test_classify_authentication():
    assert classify_error(Exception("401 invalid api key")) == FailureKind.AUTHENTICATION


def test_classify_model_not_found():
    assert classify_error(Exception("The model `foo` does not exist")) == FailureKind.MODEL_NOT_FOUND


def test_classify_timeout():
    assert classify_error(Exception("Request timed out")) == FailureKind.TIMEOUT


def test_classify_validation():
    assert classify_error(Exception("json_validate_failed: schema mismatch")) == FailureKind.VALIDATION


def test_classify_unknown_default():
    assert classify_error(Exception("something weird happened")) == FailureKind.UNKNOWN


def test_is_retryable():
    assert is_retryable(FailureKind.RATE_LIMIT) is True
    assert is_retryable(FailureKind.TIMEOUT) is True
    assert is_retryable(FailureKind.CONNECTION) is True
    assert is_retryable(FailureKind.AUTHENTICATION) is False
    assert is_retryable(FailureKind.MODEL_NOT_FOUND) is False
    assert is_retryable(FailureKind.VALIDATION) is False


def test_extract_retry_after_seconds_present():
    err = Exception("Rate limit reached. Please try again in 1.234s.")
    assert extract_retry_after_seconds(err) == 1.234


def test_extract_retry_after_seconds_absent():
    err = Exception("Rate limit reached.")
    assert extract_retry_after_seconds(err) is None


def test_call_with_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("429 too many requests")
        return "ok"

    outcome = call_with_retry(flaky, max_retries=5, base_delay=0.01, max_delay=0.02)
    assert outcome.success is True
    assert outcome.value == "ok"
    assert outcome.attempts == 3


def test_call_with_retry_does_not_retry_permanent_errors():
    calls = {"n": 0}

    def always_bad():
        calls["n"] += 1
        raise Exception("401 invalid api key")

    outcome = call_with_retry(always_bad, max_retries=5, base_delay=0.01)
    assert outcome.success is False
    assert outcome.kind == FailureKind.AUTHENTICATION
    assert calls["n"] == 1  # never retried


def test_call_with_retry_respects_max_retries():
    calls = {"n": 0}

    def always_rate_limited():
        calls["n"] += 1
        raise Exception("429 rate limit")

    outcome = call_with_retry(always_rate_limited, max_retries=3, base_delay=0.01)
    assert outcome.success is False
    assert calls["n"] == 3


def test_call_with_retry_uses_server_retry_after():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise Exception("Rate limit. Please try again in 0.05s.")
        return "ok"

    start = time.time()
    outcome = call_with_retry(flaky, max_retries=3, base_delay=5.0)  # base_delay is high on purpose
    elapsed = time.time() - start

    assert outcome.success is True
    # Should have used the short server-provided delay (0.05s), not the large base_delay.
    assert elapsed < 1.0
