"""Circuit breaker for LLM provider calls (Sorex round-2 P3.4).

When a provider (e.g. Ollama) locks up under parallel load, every LLM
call in the window times out individually.  A circuit breaker skips the
provider for a cooldown period after consecutive failures:

    qm.configure(
        provider="ollama",
        circuit_breaker=qm.CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            half_open_probes=1,
        ),
    )

Standard 3-state machine: **Closed** (normal) -> **Open** (tripped,
all calls rejected instantly) -> **Half-Open** (probing, limited
requests allowed to test recovery).

Thread safety is achieved via :class:`threading.Lock` -- the state
machine is intentionally kept simple so the critical section is tiny.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.config import LLMConfig
from quartermaster_providers.exceptions import ServiceUnavailableError
from quartermaster_providers.types import (
    NativeResponse,
    StructuredResponse,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
)


# ── Public data classes ──────────────────────────────────────────────


@dataclass(frozen=True)
class CircuitBreaker:
    """User-facing configuration object for circuit-breaker behaviour.

    Pass an instance to ``qm.configure(circuit_breaker=...)`` to wrap
    the provider with automatic failure detection and recovery probing.

    Args:
        failure_threshold: Consecutive failures before the circuit opens.
        recovery_timeout: Seconds the circuit stays open before allowing
            a half-open probe.
        half_open_probes: Number of concurrent requests allowed in the
            half-open state before deciding whether to close or re-open.
    """

    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    half_open_probes: int = 1


# ── Exception ────────────────────────────────────────────────────────


class CircuitOpenError(ServiceUnavailableError):
    """Raised when a request is rejected because the circuit is open.

    No HTTP request is fired -- the provider call is short-circuited
    immediately.
    """

    def __init__(self, provider: str | None = None):
        super().__init__(
            message=(
                "Circuit breaker is open -- provider is temporarily "
                "unavailable. Retry after the recovery timeout."
            ),
            provider=provider,
        )


# ── Thread-safe state machine ────────────────────────────────────────


class CircuitBreakerState:
    """Thread-safe 3-state circuit breaker.

    States:
        closed     -- Normal operation; failures are counted.
        open       -- Tripped; all calls are rejected until
                      *recovery_timeout* elapses.
        half_open  -- Probing; up to *half_open_probes* requests are
                      allowed.  A success closes the circuit; a
                      failure re-opens it.
    """

    def __init__(self, config: CircuitBreaker) -> None:
        self._config = config
        self._lock = threading.Lock()

        # Internal counters -- all guarded by ``_lock``.
        self._failure_count: int = 0
        self._state: Literal["closed", "open", "half_open"] = "closed"
        self._opened_at: float = 0.0  # monotonic timestamp
        self._half_open_in_flight: int = 0

    # -- Public interface --------------------------------------------------

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        """Current state of the circuit (may transition on read)."""
        with self._lock:
            self._maybe_transition()
            return self._state

    def allow_request(self) -> bool:
        """Check whether a request should be allowed.

        Returns ``True`` if the request may proceed; ``False`` if the
        circuit is open and the request should be rejected.
        """
        with self._lock:
            self._maybe_transition()

            if self._state == "closed":
                return True

            if self._state == "half_open":
                if self._half_open_in_flight < self._config.half_open_probes:
                    self._half_open_in_flight += 1
                    return True
                return False

            # state == "open"
            return False

    def record_success(self) -> None:
        """Record a successful provider call."""
        with self._lock:
            if self._state == "half_open":
                # Probe succeeded -- close the circuit.
                self._state = "closed"
                self._failure_count = 0
                self._half_open_in_flight = 0
            elif self._state == "closed":
                # Reset consecutive failure counter on success.
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed provider call."""
        with self._lock:
            if self._state == "half_open":
                # Probe failed -- re-open.
                self._state = "open"
                self._opened_at = time.monotonic()
                self._half_open_in_flight = 0
            elif self._state == "closed":
                self._failure_count += 1
                if self._failure_count >= self._config.failure_threshold:
                    self._state = "open"
                    self._opened_at = time.monotonic()

    # -- Internal helpers --------------------------------------------------

    def _maybe_transition(self) -> None:
        """Transition open -> half_open when the recovery window expires.

        Must be called under ``_lock``.
        """
        if self._state == "open":
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.recovery_timeout:
                self._state = "half_open"
                self._half_open_in_flight = 0


# ── Provider wrapper ─────────────────────────────────────────────────


class CircuitBreakerWrapper(AbstractLLMProvider):
    """Transparent wrapper that gates every provider call through a
    :class:`CircuitBreakerState`.

    Injected by :func:`~quartermaster_providers.registry.register_local`
    (or ``qm.configure``) when ``circuit_breaker=`` is passed.  All
    ``generate_*`` methods are intercepted; non-generation helpers
    (``list_models``, ``estimate_token_count``, ``prepare_tool``) are
    delegated directly to avoid tripping the breaker on metadata calls.
    """

    def __init__(
        self,
        wrapped: AbstractLLMProvider,
        breaker: CircuitBreakerState,
    ) -> None:
        self._wrapped = wrapped
        self._breaker = breaker

    # -- Passthrough helpers (no circuit-breaker gating) ----------------

    async def list_models(self) -> list[str]:
        return await self._wrapped.list_models()

    def estimate_token_count(self, text: str, model: str) -> int:
        return self._wrapped.estimate_token_count(text, model)

    def prepare_tool(self, tool: ToolDefinition) -> Any:
        return self._wrapped.prepare_tool(tool)

    async def transcribe(self, audio_path: str) -> str:
        return await self._wrapped.transcribe(audio_path)

    # -- Gated generation methods --------------------------------------

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        self._gate()
        try:
            result = await self._wrapped.generate_text_response(prompt, config)
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    async def generate_tool_parameters(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        config: LLMConfig,
    ) -> ToolCallResponse:
        self._gate()
        try:
            result = await self._wrapped.generate_tool_parameters(prompt, tools, config)
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    async def generate_native_response(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        config: LLMConfig | None = None,
    ) -> NativeResponse:
        self._gate()
        try:
            result = await self._wrapped.generate_native_response(prompt, tools, config)
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    async def generate_structured_response(
        self,
        prompt: str,
        response_schema: dict[str, Any] | type,
        config: LLMConfig,
    ) -> StructuredResponse:
        self._gate()
        try:
            result = await self._wrapped.generate_structured_response(
                prompt, response_schema, config
            )
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    # -- Internal -------------------------------------------------------

    def _gate(self) -> None:
        """Raise :class:`CircuitOpenError` if the breaker disallows."""
        if not self._breaker.allow_request():
            raise CircuitOpenError()
