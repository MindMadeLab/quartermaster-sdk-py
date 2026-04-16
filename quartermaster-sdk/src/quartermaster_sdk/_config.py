"""Module-level configuration for the v0.2.0 ergonomic API.

Callers set up their provider once at app boot:

    import quartermaster_sdk as qm
    qm.configure(
        provider="ollama",
        base_url="http://localhost:11434",    # or $OLLAMA_HOST
        default_model="gemma4:26b",           # or $QM_DEFAULT_MODEL
    )

Every subsequent ``qm.run(...)``, ``qm.instruction(...)``,
``qm.instruction_form(...)`` call picks this up automatically.  Per-call
overrides (``model="other"``, ``provider="anthropic"``) still work and
take precedence.

The design mirrors how ``openai.OpenAI()`` and ``anthropic.Anthropic()``
clients are typically used in real codebases — one construct at boot,
reused everywhere else.

**Deployment note:** the configured registry lives in module-level
globals.  Under pre-fork ASGI servers (``gunicorn -w 4``, Uvicorn with
``--workers``) each worker has its own copy — calling :func:`configure`
in the parent process before ``fork`` is visible to the children, but
calling it after fork only affects that worker.  Best practice: call
:func:`configure` inside your ASGI lifespan startup hook rather than at
module import time.  Single-process servers (``uvicorn app:api``) and
Celery workers are unaffected.
"""

from __future__ import annotations

import logging
import os

from quartermaster_providers import ProviderRegistry, register_local

logger = logging.getLogger(__name__)

# Module-level singleton state.  Threading note: writes from ``configure()``
# are not serialised — call it once at startup, not from worker threads.
_default_registry: ProviderRegistry | None = None
_default_model: str | None = None
# v0.4.0 application-level LLM timeouts. ``None`` means "leave the
# provider SDK's default behaviour untouched" — per-call overrides on
# ``qm.run(..., read_timeout=...)`` still win via the runner's
# resolution path.
_default_connect_timeout: float | None = None
_default_read_timeout: float | None = None
# v0.4.0 auto-redact PII mode (Sorex P2.2). When enabled, every
# ``user_input`` is piped through DetectPIITool + RedactPIITool before
# the LLM sees it. ``_auto_redact_policy`` controls which entity types
# are stripped; ``"all"`` means every type the detector supports.
_auto_redact_pii: bool = False
_auto_redact_policy: str = "all"


def configure(
    *,
    provider: str = "ollama",
    base_url: str | None = None,
    api_key: str | None = None,
    default_model: str | None = None,
    registry: ProviderRegistry | None = None,
    ollama_tool_protocol: str = "auto",
    timeout: float | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    auto_redact_pii: bool = False,
    auto_redact_policy: str = "all",
    telemetry: bool = False,
) -> ProviderRegistry:
    """Bind a default provider registry for subsequent ``qm.*`` calls.

    Args:
        provider: ``"ollama"``, ``"vllm"``, ``"lm-studio"``, ``"tgi"``,
            ``"localai"``, ``"llama-cpp"``, or ``"custom"``.  For
            cloud-hosted providers (openai / anthropic / groq / xai)
            register them via :class:`ProviderRegistry` directly and
            pass the result as *registry*.
        base_url: Endpoint URL.  Falls back to ``$OLLAMA_HOST`` when
            provider is ``"ollama"`` and no value is passed.
        api_key: Auth token for providers that need one.
        default_model: Model to use when a call doesn't pass ``model=``.
            Falls back to ``$QM_DEFAULT_MODEL``.
        registry: Hand in a fully-built registry instead of having
            ``configure`` build one.  Mutually exclusive with
            ``base_url`` / ``api_key``.
        ollama_tool_protocol: v0.4.0 Ollama-only transport knob for
            tool-calling requests — only consulted when
            ``provider="ollama"``.  ``"auto"`` (default) probes
            ``/api/tags`` and picks native ``/api/chat`` for models
            that advertise tool support, falling back to the OpenAI-
            compat shim for older models.  ``"native"`` forces every
            request through ``/api/chat``.  ``"openai_compat"``
            forces the pre-v0.4.0 behaviour (always
            ``/v1/chat/completions``).  Introduced to kill the
            Gemma-4 ``list_orders_v2`` / ``default_api:`` tool-name
            hallucinations the compat shim produces.

    Returns:
        The bound :class:`ProviderRegistry` — useful for tests that
        want to introspect it.
    """
    global _default_registry, _default_model
    global _default_connect_timeout, _default_read_timeout
    global _auto_redact_pii, _auto_redact_policy

    # Validate the Ollama tool-protocol knob here — we want a clear
    # ``ValueError`` at boot rather than a cryptic ``TypeError`` when
    # ``OllamaNativeProvider.__init__`` sees an unknown value.
    from quartermaster_providers.providers.ollama import VALID_TOOL_PROTOCOLS

    if ollama_tool_protocol not in VALID_TOOL_PROTOCOLS:
        raise ValueError(
            f"configure(): ollama_tool_protocol={ollama_tool_protocol!r} is invalid. "
            f"Expected one of {sorted(VALID_TOOL_PROTOCOLS)}."
        )

    # v0.4.0: resolve timeouts first so a validation failure doesn't
    # leave the module half-configured. The shortcut and split forms
    # are mutually exclusive.
    if timeout is not None and (
        connect_timeout is not None or read_timeout is not None
    ):
        raise ValueError(
            "configure(): pass timeout= OR connect_timeout/read_timeout, not both."
        )
    for label, value in (
        ("timeout", timeout),
        ("connect_timeout", connect_timeout),
        ("read_timeout", read_timeout),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"configure(): {label}= must be > 0, got {value!r}")
    if timeout is not None:
        resolved_connect: float | None = float(timeout)
        resolved_read: float | None = float(timeout)
    else:
        resolved_connect = (
            float(connect_timeout) if connect_timeout is not None else None
        )
        resolved_read = float(read_timeout) if read_timeout is not None else None

    resolved_default_model = default_model or os.environ.get("QM_DEFAULT_MODEL")

    if registry is not None:
        if base_url is not None or api_key is not None:
            raise ValueError(
                "configure(): pass either registry= OR base_url/api_key, not both."
            )
        _default_registry = registry
    else:
        # Let OllamaProvider pick up $OLLAMA_HOST when base_url is unset —
        # that's already the provider's documented behaviour.
        register_kwargs: dict[str, object] = {
            "base_url": base_url,
            "api_key": api_key,
            "default_model": resolved_default_model,
        }
        # Forward ``tool_protocol`` only to Ollama — other local
        # engines don't know the kwarg and would reject it.
        if provider == "ollama":
            register_kwargs["tool_protocol"] = ollama_tool_protocol
        _default_registry = register_local(provider, **register_kwargs)

    _default_model = resolved_default_model
    _default_connect_timeout = resolved_connect
    _default_read_timeout = resolved_read
    _auto_redact_pii = auto_redact_pii
    _auto_redact_policy = auto_redact_policy
    logger.info(
        "quartermaster_sdk configured: provider=%s default_model=%s "
        "ollama_tool_protocol=%s connect_timeout=%s read_timeout=%s",
        provider,
        resolved_default_model,
        ollama_tool_protocol if provider == "ollama" else "n/a",
        resolved_connect,
        resolved_read,
    )

    if telemetry:
        from . import telemetry as _telemetry

        _telemetry.instrument()
        logger.info("quartermaster_sdk.telemetry: instrumented from configure(telemetry=True)")

    return _default_registry


def get_default_registry() -> ProviderRegistry:
    """Return the module-level default registry or raise a helpful error."""
    if _default_registry is None:
        raise RuntimeError(
            "No default provider registry configured. Call "
            "quartermaster_sdk.configure(provider='ollama', default_model=...) "
            "once at app boot, or pass provider_registry= explicitly to run()."
        )
    return _default_registry


def get_default_model() -> str | None:
    """Return the module-level default model name, if set.

    Resolution order:

    1. The explicit ``default_model=`` passed to :func:`configure`, or
       ``$QM_DEFAULT_MODEL`` env var it picked up.
    2. The default model registered on the provider registry itself
       (e.g. via ``register_local("ollama", default_model=...)`` or
       ``registry.set_default_model("ollama", "gemma4:26b")``).
    3. ``None`` — callers then have to pass ``model=`` per call.
    """
    if _default_model is not None:
        return _default_model
    if _default_registry is not None:
        # ``ProviderRegistry.get_default_model()`` returns the fallback
        # provider's default model.  We narrow the catch so real bugs
        # (e.g. a registry that overrides the method with the wrong
        # signature) surface instead of being silently swallowed as
        # "no default".
        try:
            return _default_registry.get_default_model()
        except (AttributeError, TypeError) as exc:
            logger.warning(
                "Configured registry does not expose a usable get_default_model(): %s",
                exc,
            )
            return None
    return None


def get_default_timeouts() -> dict[str, float | None]:
    """Return the configure-time LLM call timeout defaults.

    v0.4.0 plumbing — ``_runner.py`` calls this to merge
    configure-time defaults with per-call overrides. Both keys
    default to ``None`` meaning "leave the provider SDK default
    untouched"; :func:`configure` populates them when the caller
    passed ``timeout=`` / ``connect_timeout=`` / ``read_timeout=``.
    """
    return {
        "connect_timeout": _default_connect_timeout,
        "read_timeout": _default_read_timeout,
    }


def get_auto_redact_config() -> tuple[bool, str]:
    """Return ``(auto_redact_pii, auto_redact_policy)`` from configure().

    v0.4.0 plumbing — the redaction helper reads this to decide whether
    to strip PII before the LLM sees user input.
    """
    return _auto_redact_pii, _auto_redact_policy


def reset_config() -> None:
    """Clear the configured registry — used by tests to start fresh."""
    global _default_registry, _default_model
    global _default_connect_timeout, _default_read_timeout
    global _auto_redact_pii, _auto_redact_policy
    _default_registry = None
    _default_model = None
    _default_connect_timeout = None
    _default_read_timeout = None
    _auto_redact_pii = False
    _auto_redact_policy = "all"


__all__ = [
    "configure",
    "get_auto_redact_config",
    "get_default_registry",
    "get_default_model",
    "get_default_timeouts",
    "reset_config",
]
