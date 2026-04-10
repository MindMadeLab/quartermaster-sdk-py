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


def infer_provider(model_name: str) -> str | None:
    """Infer provider name from a model name.

    Args:
        model_name: The model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").

    Returns:
        Provider name string, or None if no match found.
    """
    for pattern, provider in MODEL_PATTERNS:
        if re.match(pattern, model_name):
            return provider
    return None


class ProviderRegistry:
    """Registry for LLM provider instances and factories.

    Supports registration by name, automatic provider creation from
    stored configurations, and model-name-based provider inference.

    Example:
        registry = ProviderRegistry()
        registry.register("openai", OpenAIProvider, api_key="sk-...")
        provider = registry.get("openai")
        provider = registry.get_for_model("gpt-4o")
    """

    def __init__(self) -> None:
        self._providers: dict[str, AbstractLLMProvider] = {}
        self._factories: dict[str, tuple[type[AbstractLLMProvider], dict[str, Any]]] = {}

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

    def get_for_model(self, model_name: str) -> AbstractLLMProvider:
        """Get a provider for a model name using inference.

        Args:
            model_name: The model identifier.

        Returns:
            The provider instance.

        Raises:
            InvalidModelError: If no provider can be inferred.
            ProviderError: If the inferred provider is not registered.
        """
        provider_name = infer_provider(model_name)
        if provider_name is None:
            raise InvalidModelError(
                model_name,
                provider=f"Could not infer provider for model '{model_name}'",
            )
        return self.get(provider_name)

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

    def clear(self) -> None:
        """Remove all registrations."""
        self._providers.clear()
        self._factories.clear()


# Global default registry
_default_registry = ProviderRegistry()


def get_default_registry() -> ProviderRegistry:
    """Get the global default provider registry."""
    return _default_registry
