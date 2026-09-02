"""
Shared test fixtures.

Unit tests exercise the real retry/backoff logic in utils/retry.py
(attempt counting, error classification, giving up after max_retries),
but there's no reason for the test suite itself to sit through real
wall-clock sleeps between retries. Patch `time.sleep` to a no-op for the
retry module only, so retry *behavior* is still fully tested while the
suite stays fast.
"""

import pytest

import utils.retry as retry_module


@pytest.fixture(autouse=True)
def _skip_real_sleeps(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda seconds: None)
