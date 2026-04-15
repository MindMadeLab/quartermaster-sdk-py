"""LLM provider implementations.

This package contains concrete implementations of AbstractLLMProvider
for various LLM services — both cloud APIs and local / self-hosted engines.
"""

from quartermaster_providers.providers.openai import OpenAIProvider
from quartermaster_providers.providers.anthropic import AnthropicProvider
from quartermaster_providers.providers.google import GoogleProvider
from quartermaster_providers.providers.groq import GroqProvider
from quartermaster_providers.providers.xai import XAIProvider
from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider
from quartermaster_providers.providers.quartermaster import QuartermasterProvider
from quartermaster_providers.providers.local import (
    OllamaProvider,
    VLLMProvider,
    LMStudioProvider,
    TGIProvider,
    LocalAIProvider,
    LlamaCppProvider,
    LOCAL_PROVIDERS,
)
from quartermaster_providers.providers.ollama import (
    OllamaNativeProvider,
    model_supports_native_tools,
)

__all__ = [
    # Cloud providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "GroqProvider",
    "XAIProvider",
    "OpenAICompatibleProvider",
    "QuartermasterProvider",
    # Local / self-hosted providers
    "OllamaProvider",
    "OllamaNativeProvider",
    "VLLMProvider",
    "LMStudioProvider",
    "TGIProvider",
    "LocalAIProvider",
    "LlamaCppProvider",
    "LOCAL_PROVIDERS",
    "model_supports_native_tools",
]
