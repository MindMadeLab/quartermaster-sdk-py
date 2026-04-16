"""v0.4.0 -- circuit breaker for provider calls.

Tests cover the full 3-state machine (Closed -> Open -> Half-Open) and
the ``CircuitBreakerWrapper`` that gates ``generate_*`` calls through
the state machine.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from quartermaster_providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerWrapper,
    CircuitOpenError,
)
from quartermaster_providers.config import LLMConfig
from quartermaster_providers.exceptions import ServiceUnavailableError
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import TokenResponse


# ── Helpers ──────────────────────────────────────────────────────────


def _make_state(
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
    half_open_probes: int = 1,
) -> CircuitBreakerState:
    return CircuitBreakerState(
        CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_probes=half_open_probes,
        )
    )


# ── State machine tests ─────────────────────────────────────────────


class TestClosedState:
    def test_closed_state_allows_requests(self) -> None:
        """A fresh breaker in closed state allows all requests."""
        state = _make_state()
        assert state.state == "closed"
        assert state.allow_request() is True
        assert state.allow_request() is True
        assert state.allow_request() is True

    def test_success_resets_failure_counter(self) -> None:
        state = _make_state(failure_threshold=3)
        state.record_failure()
        state.record_failure()
        # Two failures, one more would trip.  A success resets.
        state.record_success()
        state.record_failure()
        state.record_failure()
        # Still closed -- counter was reset by the success.
        assert state.state == "closed"
        assert state.allow_request() is True


class TestOpenState:
    def test_opens_after_threshold_failures(self) -> None:
        """3 consecutive failures trip the circuit to open."""
        state = _make_state(failure_threshold=3)
        state.record_failure()
        state.record_failure()
        assert state.state == "closed"  # not yet

        state.record_failure()
        assert state.state == "open"
        assert state.allow_request() is False

    def test_open_rejects_multiple_requests(self) -> None:
        state = _make_state(failure_threshold=1)
        state.record_failure()
        assert state.state == "open"
        for _ in range(10):
            assert state.allow_request() is False


class TestHalfOpenState:
    def test_half_open_after_recovery_timeout(self) -> None:
        """After recovery_timeout seconds the circuit transitions to
        half_open and allows one probe."""
        state = _make_state(failure_threshold=1, recovery_timeout=0.05)
        state.record_failure()
        assert state.state == "open"

        time.sleep(0.1)

        assert state.state == "half_open"
        assert state.allow_request() is True

    def test_half_open_success_closes_circuit(self) -> None:
        """A successful probe in half_open closes the circuit."""
        state = _make_state(failure_threshold=1, recovery_timeout=0.05)
        state.record_failure()
        time.sleep(0.1)

        assert state.state == "half_open"
        assert state.allow_request() is True
        state.record_success()
        assert state.state == "closed"
        # Normal traffic resumes.
        assert state.allow_request() is True

    def test_half_open_failure_reopens(self) -> None:
        """A failed probe in half_open re-opens the circuit."""
        state = _make_state(failure_threshold=1, recovery_timeout=0.05)
        state.record_failure()
        time.sleep(0.1)

        assert state.state == "half_open"
        assert state.allow_request() is True
        state.record_failure()
        assert state.state == "open"
        assert state.allow_request() is False

    def test_half_open_limits_probes(self) -> None:
        """Only half_open_probes requests are allowed in half_open."""
        state = _make_state(failure_threshold=1, recovery_timeout=0.05, half_open_probes=2)
        state.record_failure()
        time.sleep(0.1)

        assert state.state == "half_open"
        assert state.allow_request() is True
        assert state.allow_request() is True
        # Third request is rejected.
        assert state.allow_request() is False


class TestThreadSafety:
    def test_thread_safety(self) -> None:
        """Concurrent calls from 10 threads produce no race conditions.

        We hammer the state machine with alternating successes and
        failures from 10 threads and assert the state is always one
        of the three valid values.
        """
        state = _make_state(failure_threshold=5, recovery_timeout=0.02)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(100):
                    state.allow_request()
                    if i % 3 == 0:
                        state.record_failure()
                    else:
                        state.record_success()
                    # Verify the state is always valid.
                    assert state.state in ("closed", "open", "half_open")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread safety errors: {errors}"


# ── Wrapper tests ────────────────────────────────────────────────────


class TestCircuitBreakerWrapper:
    @pytest.fixture()
    def mock_provider(self) -> MockProvider:
        return MockProvider(responses=[TokenResponse(content="ok", stop_reason="end_turn")])

    @pytest.fixture()
    def config(self) -> LLMConfig:
        return LLMConfig(model="mock-model-1", provider="mock")

    def test_wrapper_delegates_to_inner_provider(
        self, mock_provider: MockProvider, config: LLMConfig
    ) -> None:
        """Success path calls through to the wrapped provider and
        records a success."""
        breaker = _make_state(failure_threshold=3)
        wrapper = CircuitBreakerWrapper(mock_provider, breaker)

        result = asyncio.get_event_loop().run_until_complete(
            wrapper.generate_text_response("hello", config)
        )

        assert result.content == "ok"
        assert mock_provider.call_count == 1
        assert breaker.state == "closed"

    def test_open_raises_circuit_open_error(
        self, mock_provider: MockProvider, config: LLMConfig
    ) -> None:
        """When the circuit is open, CircuitOpenError is raised
        immediately without calling the inner provider."""
        breaker = _make_state(failure_threshold=1)
        wrapper = CircuitBreakerWrapper(mock_provider, breaker)

        # Trip the circuit.
        breaker.record_failure()
        assert breaker.state == "open"

        with pytest.raises(CircuitOpenError):
            asyncio.get_event_loop().run_until_complete(
                wrapper.generate_text_response("hello", config)
            )

        # Inner provider was never called.
        assert mock_provider.call_count == 0

    def test_circuit_open_error_is_service_unavailable(self) -> None:
        """CircuitOpenError is a subclass of ServiceUnavailableError."""
        assert issubclass(CircuitOpenError, ServiceUnavailableError)

    def test_failure_recorded_on_provider_exception(self, config: LLMConfig) -> None:
        """When the inner provider raises, the wrapper records a failure."""

        class FailingProvider(MockProvider):
            async def generate_text_response(self, prompt, config):
                raise RuntimeError("provider down")

        breaker = _make_state(failure_threshold=3)
        wrapper = CircuitBreakerWrapper(FailingProvider(), breaker)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                asyncio.get_event_loop().run_until_complete(
                    wrapper.generate_text_response("hello", config)
                )

        assert breaker.state == "open"

    def test_passthrough_methods_not_gated(self, mock_provider: MockProvider) -> None:
        """list_models, estimate_token_count, prepare_tool are not
        gated by the circuit breaker."""
        breaker = _make_state(failure_threshold=1)
        breaker.record_failure()
        assert breaker.state == "open"

        wrapper = CircuitBreakerWrapper(mock_provider, breaker)

        # These should still work even with an open circuit.
        models = asyncio.get_event_loop().run_until_complete(wrapper.list_models())
        assert "mock-model-1" in models

        count = wrapper.estimate_token_count("hello world", "mock-model-1")
        assert count == 2
