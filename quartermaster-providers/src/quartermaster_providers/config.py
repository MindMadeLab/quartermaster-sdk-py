"""LLM configuration and request parameters.

This module defines the LLMConfig class that unifies configuration across
all LLM providers, allowing consistent parameter passing regardless of which
provider is used.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM requests.

    Attributes:
        model: The model identifier for the provider (e.g., 'gpt-4', 'claude-3-opus').
        provider: Provider identifier ('openai', 'anthropic', 'google', 'groq', 'xai').
        stream: Whether to stream the response token-by-token.
        temperature: Sampling temperature (0.0-2.0). Lower = deterministic, higher = creative.
        system_message: Optional system prompt to set model behavior.
        max_input_tokens: Maximum tokens in the input prompt.
        max_output_tokens: Maximum tokens in the response.
        max_messages: Maximum number of messages to include in conversation context.
        vision: Whether the request includes vision/image understanding.
        images: Optional list of ``(base64_data, mime_type)`` pairs to send
            alongside the prompt for vision-capable models. ``base64_data``
            is the raw image bytes encoded as ASCII base64 (no ``data:``
            URI prefix); ``mime_type`` is e.g. ``"image/jpeg"`` /
            ``"image/png"`` / ``"image/webp"``. Populated by the v0.3.0
            engine path that reads ``flow_memory["__user_images__"]``;
            providers that support vision consume this list when building
            the request payload. Empty/``None`` means a text-only request.
        thinking_enabled: Whether to enable extended thinking mode (e.g., Claude thinking).
        thinking_budget: Maximum tokens allowed for thinking/reasoning.
        top_p: Nucleus sampling parameter (alternative to temperature).
        top_k: Top-k sampling parameter.
        frequency_penalty: Penalty for repeating tokens (OpenAI).
        presence_penalty: Penalty for new tokens (OpenAI).
        connect_timeout: Connect-phase timeout in seconds — fail fast if
            the provider endpoint is unreachable. ``None`` means use the
            SDK's underlying HTTP client default. Added in v0.4.0.
        read_timeout: Read-phase timeout in seconds — ceiling for waiting
            on a single streaming token / complete response. ``None``
            means use the SDK's underlying HTTP client default. Added
            in v0.4.0 so Celery / worker tasks no longer depend on
            blunt ``CELERY_TASK_TIME_LIMIT`` kills when an Ollama
            instance wedges mid-stream.
    """

    model: str
    provider: str
    stream: bool = False
    temperature: float = 0.7
    system_message: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_messages: int | None = None
    vision: bool = False
    images: list[tuple[str, str]] = field(default_factory=list)
    thinking_enabled: bool = False
    thinking_budget: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # v0.4.0 timeouts — threaded through to each provider's HTTP client
    # when the SDK's ``qm.configure(timeout=/connect_timeout=/read_timeout=)``
    # or per-call ``qm.run(..., read_timeout=)`` kwargs are used.
    # ``None`` on both leaves the provider SDK's own default behaviour
    # untouched (backwards-compat with v0.3.x callers).
    connect_timeout: float | None = None
    read_timeout: float | None = None
    # v0.6.0 — pass-through for provider-specific OpenAI-compat body fields
    # that the SDK doesn't otherwise model. The dict is spliced into the
    # outgoing ``chat.completions.create(..., extra_body=<dict>)`` call
    # (supported by the openai Python SDK as a generic escape hatch).
    #
    # Primary motivator: Gemma-4's ``chat_template_kwargs`` knob for
    # toggling <thinking> blocks per-request:
    #   extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    # Also handy for vLLM-specific sampling params (``top_k``,
    # ``repetition_penalty``) that don't have a first-class field here.
    #
    # The field is provider-agnostic at the ``LLMConfig`` layer — each
    # provider decides whether / how to forward it. OpenAI +
    # OpenAI-compatible providers splice it into the request; Anthropic,
    # Google, Groq, xAI ignore it (or will once they grow bespoke
    # pass-through slots of their own).
    extra_body: dict[str, Any] | None = None

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.model:
            raise ValueError("model is required")

        if not self.provider:
            raise ValueError("provider is required")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")

        if self.max_input_tokens is not None and self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1")

        if self.thinking_budget is not None and self.thinking_budget < 1:
            raise ValueError("thinking_budget must be >= 1")

        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")

        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")

        if self.frequency_penalty is not None and not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be between -2.0 and 2.0")

        if self.presence_penalty is not None and not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be between -2.0 and 2.0")

        if self.connect_timeout is not None and self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be > 0")

        if self.read_timeout is not None and self.read_timeout <= 0:
            raise ValueError("read_timeout must be > 0")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LLMConfig":
        """Create LLMConfig from a dictionary.

        Args:
            config_dict: Dictionary with LLMConfig fields.

        Returns:
            LLMConfig instance.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = {"model", "provider"}
        if not required_fields.issubset(config_dict.keys()):
            missing = required_fields - set(config_dict.keys())
            raise ValueError(f"Missing required fields: {missing}")

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "model": self.model,
            "provider": self.provider,
            "stream": self.stream,
            "temperature": self.temperature,
            "system_message": self.system_message,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_messages": self.max_messages,
            "vision": self.vision,
            "images": list(self.images),
            "thinking_enabled": self.thinking_enabled,
            "thinking_budget": self.thinking_budget,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
        }
