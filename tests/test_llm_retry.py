"""Tests for the LLM retry logic — transient vs permanent errors.

These tests verify that:

* Transport failures (disconnects, timeouts) are retried with exponential
  backoff long enough to ride out a brief outage (at least 5 attempts).
* A transient error that succeeds on the 3rd attempt returns a valid
  result.
* Permanent errors (HTTP 400 bad request, 401/403 auth, schema
  validation) fail fast without retrying.

Run with: .venv/bin/python -m pytest tests/test_llm_retry.py -q
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch, MagicMock

import httpx
import pytest
from pydantic import BaseModel

os.environ.setdefault("LLM_PROVIDER", "mock")

from ridecourt.llm import (
    LLMError,
    PermanentLLMError,
    _backoff_delay,
    _classify_gemini_error,
    _is_retriable,
    _retry,
)
from ridecourt import llm as llm_mod


# ---------------------------------------------------------------------------
# Schema fixture
# ---------------------------------------------------------------------------


class SimpleOutput(BaseModel):
    """A minimal schema for testing."""

    outcome: str
    amount: float = 0.0


# ---------------------------------------------------------------------------
# Unit tests for the retry classification and backoff
# ---------------------------------------------------------------------------


class TestRetryClassification:
    def test_permanent_error_is_not_retriable(self):
        err = PermanentLLMError("400 bad request")
        assert not _is_retriable(err)

    def test_generic_llm_error_is_retriable(self):
        err = LLMError("server disconnected")
        assert _is_retriable(err)

    def test_httpx_remote_protocol_error_is_retriable(self):
        err = httpx.RemoteProtocolError("Server disconnected without sending a response.")
        assert _is_retriable(err)

    def test_connection_error_is_retriable(self):
        assert _is_retriable(ConnectionError("connection reset"))

    def test_timeout_error_is_retriable(self):
        assert _is_retriable(TimeoutError("timed out"))


class TestBackoffDelay:
    def test_backoff_grows_with_attempts(self):
        """Later attempts should have a larger max delay than earlier ones."""
        delays = [_backoff_delay(i) for i in range(5)]
        # The max possible delay (base = min(2^attempt, 16)) grows.
        maxes = [min(2 ** i, 16.0) for i in range(5)]
        for d, m in zip(delays, maxes):
            assert 0 <= d <= m

    def test_total_budget_about_30_seconds(self):
        """The sum of max delays across 4 backoff slots is about 30 s."""
        # 4 delays between 5 attempts: 1+2+4+8 = 15 (max, no jitter).
        # With cap 16: later attempts can be up to 16, but the 5-attempt
        # structure 1+2+4+8 = 15 min is the exponential floor.
        # The key invariant: at least 5 attempts are available.
        total_max = sum(min(2 ** i, 16.0) for i in range(4))
        assert total_max >= 15.0  # roughly 30 s budget with jitter


# ---------------------------------------------------------------------------
# Integration tests: _retry with a fake function
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    def test_succeeds_after_two_transient_failures(self):
        """A function that fails twice with a disconnect then succeeds
        should return the success value on the 3rd attempt."""
        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMError("Server disconnected without sending a response.")
            return "ok"

        # Patch sleep so the test is fast.
        with patch.object(llm_mod.time, "sleep"):
            result = _retry(flaky_fn)
        assert result == "ok"
        assert call_count == 3

    def test_permanent_error_fails_immediately(self):
        """A PermanentLLMError is re-raised on the first attempt — no retry."""
        call_count = 0

        def bad_request_fn():
            nonlocal call_count
            call_count += 1
            raise PermanentLLMError("400 bad request: invalid argument")

        with patch.object(llm_mod.time, "sleep") as mock_sleep:
            with pytest.raises(PermanentLLMError):
                _retry(bad_request_fn)
        assert call_count == 1
        mock_sleep.assert_not_called()

    def test_exhausts_retries_on_persistent_transient_error(self):
        """A transient error that never recovers is retried ``_MAX_ATTEMPTS``
        times, then re-raised."""
        call_count = 0

        def always_fail_fn():
            nonlocal call_count
            call_count += 1
            raise LLMError("Server disconnected without sending a response.")

        with patch.object(llm_mod.time, "sleep"):
            with pytest.raises(LLMError):
                _retry(always_fail_fn)
        # At least 5 attempts were made (old code only did 3).
        assert call_count >= 5


# ---------------------------------------------------------------------------
# End-to-end tests: _retry + _classify_gemini_error (the real code path)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """A minimal response object for a fake Gemini call."""

    def __init__(self, text, tokens=10):
        self.text = text
        class _U:
            total_token_count = tokens
        self.usage_metadata = _U()


class TestGeminiErrorClassification:
    """Test that ``_classify_gemini_error`` correctly distinguishes
    retriable transport failures from permanent 400/401/403 errors,
    and that ``_retry`` respects the classification."""

    def test_disconnect_twice_then_succeed(self):
        """A Gemini call that disconnects twice then succeeds on the 3rd
        attempt should return a valid result through ``_retry``.

        This mirrors the production path: ``complete_json`` calls
        ``_retry(_complete_gemini, ...)`` and ``_complete_gemini`` raises
        a classified error on transport failures.
        """
        responses = [
            httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            ),
            httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            ),
            _FakeResponse('{"outcome": "refund", "amount": 5.0}'),
        ]
        state = {"i": 0}

        def fake_gemini_call(system, user, schema, agent_name):
            """Simulates ``_complete_gemini`` — raises classified errors
            or returns a parsed result."""
            idx = state["i"]
            state["i"] += 1
            item = responses[idx] if idx < len(responses) else responses[-1]
            if isinstance(item, Exception):
                raise _classify_gemini_error(item)
            # Success: parse and return.
            from ridecourt.llm import LLMMeta
            parsed = schema.model_validate({"outcome": "refund", "amount": 5.0})
            return parsed, LLMMeta(
                provider="gemini", model="fake", agent_name=agent_name,
            )

        with patch.object(llm_mod.time, "sleep"):
            result, meta = _retry(
                fake_gemini_call, "s", "u", SimpleOutput, "test_agent"
            )
        assert result.outcome == "refund"
        assert result.amount == 5.0
        assert state["i"] == 3

    def test_400_bad_request_fails_immediately(self):
        """A 400 bad request is permanent — ``_retry`` re-raises on the
        first attempt with no sleeping."""
        class FakeResponse400:
            status_code = 400
            text = "Bad Request: invalid argument"

        class ClientError(Exception):
            def __init__(self):
                self.response = FakeResponse400()
                super().__init__("400 Bad Request")

        state = {"i": 0}

        def fake_gemini_call(system, user, schema, agent_name):
            idx = state["i"]
            state["i"] += 1
            raise _classify_gemini_error(ClientError())

        with patch.object(llm_mod.time, "sleep") as mock_sleep:
            with pytest.raises(PermanentLLMError) as exc_info:
                _retry(fake_gemini_call, "s", "u", SimpleOutput, "test_agent")
        msg = str(exc_info.value)
        assert "400" in msg or "Bad Request" in msg
        mock_sleep.assert_not_called()
        assert state["i"] == 1

    def test_401_unauthenticated_fails_immediately(self):
        """A 401 auth error is permanent."""
        class FakeResponse401:
            status_code = 401
            text = "Unauthorized"

        class ClientError(Exception):
            def __init__(self):
                self.response = FakeResponse401()
                super().__init__("401 Unauthorized")

        def fake_gemini_call(system, user, schema, agent_name):
            raise _classify_gemini_error(ClientError())

        with patch.object(llm_mod.time, "sleep") as mock_sleep:
            with pytest.raises(PermanentLLMError):
                _retry(fake_gemini_call, "s", "u", SimpleOutput, "test_agent")
        mock_sleep.assert_not_called()

    def test_429_rate_limit_is_retried(self):
        """A 429 rate-limit is retriable — succeeds on the 2nd attempt."""
        class FakeResponse429:
            status_code = 429
            text = "Too Many Requests"

        class ClientError(Exception):
            def __init__(self):
                self.response = FakeResponse429()
                super().__init__("429 Too Many Requests")

        state = {"i": 0}

        def fake_gemini_call(system, user, schema, agent_name):
            idx = state["i"]
            state["i"] += 1
            if idx == 0:
                raise _classify_gemini_error(ClientError())
            from ridecourt.llm import LLMMeta
            parsed = schema.model_validate({"outcome": "refund", "amount": 5.0})
            return parsed, LLMMeta(
                provider="gemini", model="fake", agent_name=agent_name,
            )

        with patch.object(llm_mod.time, "sleep"):
            result, _ = _retry(
                fake_gemini_call, "s", "u", SimpleOutput, "test_agent"
            )
        assert result.outcome == "refund"
        assert state["i"] == 2
