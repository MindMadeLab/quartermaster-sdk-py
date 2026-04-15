"""Provider registry and model resolution.

Manages provider registration, lookup, and automatic model-to-provider
inference based on model name patterns.
"""

from __future__ import annotations

import re
from typing import Any

from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.exceptions import InvalidModelError, ProviderError

# Model name patterns mapped to provider names
MODEL_PATTERNS: list[tuple[str, str]] = [
    # OpenAI
    (r"^gpt-", "openai"),
    (r"^o[13]-", "openai"),
    (r"^o[13]$", "openai"),
    (r"^chatgpt-", "openai"),
    (r"^dall-e", "openai"),
    (r"^whisper", "openai"),
    (r"^tts-", "openai"),
    # Anthropic
    (r"^claude-", "anthropic"),
    # Google
    (r"^gemini-", "google"),
    (r"^gemma-", "google"),
    # Groq
    (r"^llama-", "groq"),
    (r"^mixtral-", "groq"),
    # xAI
    (r"^grok-", "xai"),
]


def infer_provider(
    model_name: str,
    extra_patterns: list[tuple[str, str]] | None = None,
) -> str | None:
    """Infer provider name from a model name.

    Checks *extra_patterns* first (instance-level custom routes) then
    falls back to the global ``MODEL_PATTERNS``.

    Args:
        model_name: The model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        extra_patterns: Additional ``(regex, provider_name)`` pairs to
            check **before** the built-in list.

    Returns:
        Provider name string, or None if no match found.
    """
    for pattern, provider in extra_patterns or []:
        if re.match(pattern, model_name):
            return provider
    for pattern, provider in MODEL_PATTERNS:
        if re.match(pattern, model_name):
            return provider
    return None


class ProviderRegistry:
    """Registry for LLM provider instances and factories.

    Supports registration by name, automatic provider creation from
    stored configurations, and model-name-based provider inference.

    When ``QUARTERMASTER_API_KEY`` is set in the environment, a
    ``QuartermasterProvider`` is automatically registered and used as the
    fallback for all model inference — so a single API key gives access
    to every supported model.

    Example:
        registry = ProviderRegistry()
        registry.register("openai", OpenAIProvider, api_key="sk-...")
        provider = registry.get("openai")
        provider = registry.get_for_model("gpt-4o")
    """

    def __init__(self, auto_configure: bool = True) -> None:
        self._providers: dict[str, AbstractLLMProvider] = {}
        self._factories: dict[str, tuple[type[AbstractLLMProvider], dict[str, Any]]] = {}
        self._custom_patterns: list[tuple[str, str]] = []
        self._default_provider: str | None = None
        self._default_models: dict[str, str] = {}
        self._auto_configured = False
        if auto_configure:
            self._auto_configure()

    def _auto_configure(self) -> None:
        """Auto-register QuartermasterProvider when QUARTERMASTER_API_KEY is set."""
        import os

        api_key = os.environ.get("QUARTERMASTER_API_KEY")
        if api_key:
            try:
                from quartermaster_providers.providers.quartermaster import (
                    QuartermasterProvider,
                )

                self._factories["quartermaster"] = (QuartermasterProvider, {"api_key": api_key})
                self._auto_configured = True
            except ImportError:
                pass

    def register(
        self,
        name: str,
        provider_cls: type[AbstractLLMProvider],
        **kwargs: Any,
    ) -> None:
        """Register a provider class with constructor arguments.

        The provider will be lazily instantiated on first `get()` call.

        Args:
            name: Provider name (e.g., "openai", "anthropic", "my-ollama").
            provider_cls: The provider class.
            **kwargs: Constructor arguments for the provider.
        """
        self._factories[name] = (provider_cls, kwargs)
        # Clear cached instance so it gets recreated with new config
        self._providers.pop(name, None)

    def register_instance(self, name: str, provider: AbstractLLMProvider) -> None:
        """Register a pre-created provider instance.

        Args:
            name: Provider name.
            provider: The provider instance.
        """
        self._providers[name] = provider

    def get(self, name: str) -> AbstractLLMProvider:
        """Get a provider by name, creating it if necessary.

        Args:
            name: Provider name.

        Returns:
            The provider instance.

        Raises:
            ProviderError: If provider is not registered.
        """
        if name in self._providers:
            return self._providers[name]

        if name in self._factories:
            cls, kwargs = self._factories[name]
            instance = cls(**kwargs)
            self._providers[name] = instance
            return instance

        raise ProviderError(
            f"Provider '{name}' is not registered. Available: {', '.join(self.list_providers())}"
        )

    def register_local(
        self,
        engine: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        name: str | None = None,
        models: list[str] | None = None,
        default: bool = False,
        default_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """One-liner registration for local / self-hosted LLM engines.

        Supported engines: ``"ollama"``, ``"vllm"``, ``"lm-studio"``,
        ``"tgi"``, ``"localai"``, ``"llama-cpp"``.

        Args:
            engine: Engine shorthand (see above) or ``"custom"`` with
                *base_url* to use the generic ``OpenAICompatibleProvider``.
            base_url: Override the default endpoint URL.
            api_key: Override the default API key.
            name: Provider name in the registry (defaults to *engine*).
            models: Optional list of model-name regex patterns to route
                to this provider automatically.  For example
                ``["llama3.*", "codellama.*"]`` will make
                ``get_for_model("llama3.1:70b")`` resolve here.
            default: If ``True``, set this as the default fallback
                provider for any model that can't be resolved.
            default_model: Optional default model to associate with this
                provider — engine-level helpers (``FlowRunner`` /
                ``run_graph``) use it when a node doesn't pin its own
                ``llm_model``.  Implies ``default=True`` so unrouted
                model names also resolve here, and registers a literal
                pattern for the model name itself.
            **kwargs: Extra keyword arguments forwarded to the provider
                constructor.

        Example::

            registry = ProviderRegistry()

            # Simplest — Ollama with defaults
            registry.register_local("ollama")

            # vLLM on a remote GPU box, route specific models to it
            registry.register_local(
                "vllm",
                base_url="http://gpu-box:8000/v1",
                models=["llama3.*", "mistral.*", "codellama.*"],
            )

            # Completely custom endpoint
            registry.register_local(
                "custom",
                base_url="http://my-server:9000/v1",
                name="my-infra",
                api_key="secret",
                default=True,
            )
        """
        from quartermaster_providers.providers.local import (
            LOCAL_PROVIDERS,
            _normalize_openai_compat_url,
        )
        from quartermaster_providers.providers.openai_compat import (
            OpenAICompatibleProvider,
        )

        provider_name = name or engine

        if engine == "custom":
            if not base_url:
                raise ProviderError("engine='custom' requires a base_url argument.")
            ctor_kwargs: dict[str, Any] = {
                "base_url": _normalize_openai_compat_url(base_url),
                **kwargs,
            }
            if api_key:
                ctor_kwargs["api_key"] = api_key
            ctor_kwargs.setdefault("provider_name", provider_name)
            self._factories[provider_name] = (OpenAICompatibleProvider, ctor_kwargs)
        elif engine in LOCAL_PROVIDERS:
            cls = LOCAL_PROVIDERS[engine]
            ctor_kwargs = {**kwargs}
            if base_url:
                ctor_kwargs["base_url"] = _normalize_openai_compat_url(base_url)
            if api_key:
                ctor_kwargs["api_key"] = api_key
            # Pass default_model through so providers that expose a sync
            # facade (currently OllamaProvider.chat()) can fall back to
            # it when callers omit ``model=`` per call.  Only forwarded
            # when the provider class accepts it, to stay backwards-
            # compatible with engines that don't.
            if default_model and "default_model" in cls.__init__.__code__.co_varnames:
                ctor_kwargs["default_model"] = default_model
            self._factories[provider_name] = (cls, ctor_kwargs)
        else:
            raise ProviderError(
                f"Unknown engine '{engine}'. "
                f"Available: {', '.join(sorted(LOCAL_PROVIDERS))} or 'custom'."
            )

        # Clear cached instance
        self._providers.pop(provider_name, None)

        # Register model-name routing patterns
        if models:
            for pattern in models:
                self.add_model_pattern(pattern, provider_name)

        if default_model:
            # Route the literal model name to this provider and remember it
            # so consumers (FlowRunner et al.) can resolve "no model
            # specified" → the user's intended default for this engine.
            self.add_model_pattern(re.escape(default_model), provider_name)
            self._default_models[provider_name] = default_model
            # default_model implies "use me when nothing else matches"
            self.set_default_provider(provider_name)

        if default:
            self.set_default_provider(provider_name)

    def add_model_pattern(self, pattern: str, provider_name: str) -> None:
        """Add a custom model-name → provider routing rule.

        Custom patterns are checked **before** the built-in patterns,
        so they can override defaults.  For example::

            registry.add_model_pattern(r"llama3.*", "ollama")

        makes ``get_for_model("llama3.1:70b")`` resolve to ``"ollama"``
        instead of ``"groq"`` (the built-in default for ``llama-``).

        Args:
            pattern: Regex pattern to match against model names.
            provider_name: Provider name to resolve to.
        """
        self._custom_patterns.append((pattern, provider_name))

    def set_default_provider(self, name: str) -> None:
        """Set a provider as the catch-all fallback.

        When ``get_for_model()`` can't resolve a model name, it will
        use this provider instead of raising an error.

        Args:
            name: Registered provider name.
        """
        self._default_provider = name

    def get_for_model(self, model_name: str) -> AbstractLLMProvider:
        """Get a provider for a model name using inference.

        Resolution order:

        1. Custom patterns (added via ``add_model_pattern()`` /
           ``register_local(..., models=[...])``) — checked first.
        2. Built-in ``MODEL_PATTERNS`` (OpenAI, Anthropic, Google, etc.).
        3. Default provider (set via ``set_default_provider()`` /
           ``register_local(..., default=True)``).
        4. ``QuartermasterProvider`` fallback (via ``QUARTERMASTER_API_KEY``).

        Args:
            model_name: The model identifier.

        Returns:
            The provider instance.

        Raises:
            InvalidModelError: If no provider can be inferred.
            ProviderError: If the inferred provider is not registered.
        """
        provider_name = infer_provider(model_name, extra_patterns=self._custom_patterns)
        if provider_name is None:
            # Try default provider
            if self._default_provider and self.is_registered(self._default_provider):
                return self.get(self._default_provider)
            # If Quartermaster is available, use it as a catch-all
            if self.is_registered("quartermaster"):
                return self.get("quartermaster")
            raise InvalidModelError(
                model_name,
                provider=f"Could not infer provider for model '{model_name}'",
            )
        # If the inferred provider is registered, use it directly
        if self.is_registered(provider_name):
            return self.get(provider_name)
        # Fall back to default provider
        if self._default_provider and self.is_registered(self._default_provider):
            return self.get(self._default_provider)
        # Fall back to Quartermaster if available
        if self.is_registered("quartermaster"):
            return self.get("quartermaster")
        raise ProviderError(
            f"Provider '{provider_name}' (for model '{model_name}') is not registered "
            f"and no Quartermaster API key is configured. "
            f"Either register the provider or set QUARTERMASTER_API_KEY."
        )

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        names = set(self._providers.keys()) | set(self._factories.keys())
        return sorted(names)

    def is_registered(self, name: str) -> bool:
        """Check if a provider is registered."""
        return name in self._providers or name in self._factories

    def unregister(self, name: str) -> None:
        """Remove a provider registration."""
        self._providers.pop(name, None)
        self._factories.pop(name, None)
        self._default_models.pop(name, None)

    def clear(self) -> None:
        """Remove all registrations."""
        self._providers.clear()
        self._factories.clear()
        self._custom_patterns.clear()
        self._default_provider = None
        self._default_models.clear()

    def get_default_model(self, provider_name: str | None = None) -> str | None:
        """Return the default model for a provider (or the registry's default).

        When *provider_name* is omitted, the registry's default-fallback
        provider's default model is used (set via
        ``register_local(default_model=...)`` or ``set_default_model()``).
        """
        if provider_name is None:
            provider_name = self._default_provider
        if not provider_name:
            return None
        return self._default_models.get(provider_name)

    def set_default_model(self, provider_name: str, model: str) -> None:
        """Associate a default model with a provider name.

        This does not affect routing — it just records the model so engine-
        level helpers can pick it when a node leaves ``llm_model`` blank.
        """
        self._default_models[provider_name] = model

    @property
    def default_provider(self) -> str | None:
        """Name of the catch-all fallback provider, if one is configured."""
        return self._default_provider


# Global default registry
_default_registry = ProviderRegistry()


def get_default_registry() -> ProviderRegistry:
    """Get the global default provider registry."""
    return _default_registry


def register_local(
    engine: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    name: str | None = None,
    models: list[str] | None = None,
    default: bool = True,
    default_model: str | None = None,
    registry: ProviderRegistry | None = None,
    **kwargs: Any,
) -> ProviderRegistry:
    """Module-level convenience: build a registry with one local engine.

    The simplest possible path from "I have Ollama running" to "I have a
    ``ProviderRegistry`` ready to hand to the engine"::

        from quartermaster_providers import register_local

        provider_registry = register_local(
            "ollama",
            base_url="http://host.docker.internal:11434",
            default_model="gemma4:26b",
        )

    *registry* lets callers extend an existing registry instead of
    creating a fresh one.  All other arguments are forwarded to
    :meth:`ProviderRegistry.register_local`.
    """
    reg = registry if registry is not None else ProviderRegistry(auto_configure=False)
    reg.register_local(
        engine,
        base_url=base_url,
        api_key=api_key,
        name=name,
        models=models,
        default=default,
        default_model=default_model,
        **kwargs,
    )
    return reg
