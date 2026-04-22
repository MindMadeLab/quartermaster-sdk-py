"""Unified multi-LLM provider abstraction.

quartermaster-providers offers a single, consistent interface to interact with multiple
Large Language Models from different providers (OpenAI, Anthropic, Google, Groq,
xAI, and custom implementations).

Example:
    from quartermaster_providers import LLMConfig, ProviderRegistry
    from quartermaster_providers.providers import OpenAIProvider

    registry = ProviderRegistry()
    registry.register("openai", OpenAIProvider, api_key="sk-...")
    provider = registry.get_for_model("gpt-4o")

    config = LLMConfig(model="gpt-4o", provider="openai")
    response = await provider.generate_text_response("Hello!", config)
"""

from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerWrapper,
    CircuitOpenError,
)
from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.local import ChatResult
from quartermaster_providers.exceptions import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidModelError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)
from quartermaster_providers.registry import (
    ProviderRegistry,
    get_default_registry,
    infer_provider,
    register_local,
)
from quartermaster_providers.types import (
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

__version__ = "0.4.10"
__author__ = "MindMade"

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
    "register_local",
    # Sync chat shim result type
    "ChatResult",
    # Circuit breaker (v0.4.0)
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerWrapper",
    "CircuitOpenError",
]
