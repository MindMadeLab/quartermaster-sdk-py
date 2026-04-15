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


def configure(
    *,
    provider: str = "ollama",
    base_url: str | None = None,
    api_key: str | None = None,
    default_model: str | None = None,
    registry: ProviderRegistry | None = None,
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

    Returns:
        The bound :class:`ProviderRegistry` — useful for tests that
        want to introspect it.
    """
    global _default_registry, _default_model

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
        _default_registry = register_local(
            provider,
            base_url=base_url,
            api_key=api_key,
            default_model=resolved_default_model,
        )

    _default_model = resolved_default_model
    logger.info(
        "quartermaster_sdk configured: provider=%s default_model=%s",
        provider,
        resolved_default_model,
    )
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


def reset_config() -> None:
    """Clear the configured registry — used by tests to start fresh."""
    global _default_registry, _default_model
    _default_registry = None
    _default_model = None


__all__ = [
    "configure",
    "get_default_registry",
    "get_default_model",
    "reset_config",
]
