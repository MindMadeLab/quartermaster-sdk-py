"""Unified multi-LLM provider abstraction.

qm-providers offers a single, consistent interface to interact with multiple
Large Language Models from different providers (OpenAI, Anthropic, Google, Groq,
xAI, and custom implementations).

Example:
    from qm_providers import LLMConfig, ProviderRegistry
    from qm_providers.providers import OpenAIProvider

    registry = ProviderRegistry()
    registry.register("openai", OpenAIProvider, api_key="sk-...")
    provider = registry.get_for_model("gpt-4o")

    config = LLMConfig(model="gpt-4o", provider="openai")
    response = await provider.generate_text_response("Hello!", config)
"""

from qm_providers.config import LLMConfig
from qm_providers.types import (
    Message,
    MessageHistory,
    NativeResponse,
    StructuredResponse,
    ThinkingResponse,
    TokenResponse,
    TokenUsage,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
)
from qm_providers.base import AbstractLLMProvider
from qm_providers.exceptions import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidModelError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)
from qm_providers.registry import ProviderRegistry, infer_provider, get_default_registry

__version__ = "0.1.0"
__author__ = "Quartermaster AI"

__all__ = [
    # Configuration
    "LLMConfig",
    # Types
    "TokenResponse",
    "ThinkingResponse",
    "TokenUsage",
    "ToolCall",
    "ToolCallResponse",
    "StructuredResponse",
    "NativeResponse",
    "ToolDefinition",
    "Message",
    "MessageHistory",
    # Base class
    "AbstractLLMProvider",
    # Exceptions
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidModelError",
    "InvalidRequestError",
    "ContentFilterError",
    "ContextLengthError",
    "ServiceUnavailableError",
    # Registry
    "ProviderRegistry",
    "infer_provider",
    "get_default_registry",
]
