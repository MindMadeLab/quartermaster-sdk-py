"""LLM configuration and request parameters.

This module defines the LLMConfig class that unifies configuration across
all LLM providers, allowing consistent parameter passing regardless of which
provider is used.
"""

from dataclasses import dataclass


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
        thinking_enabled: Whether to enable extended thinking mode (e.g., Claude thinking).
        thinking_budget: Maximum tokens allowed for thinking/reasoning.
        top_p: Nucleus sampling parameter (alternative to temperature).
        top_k: Top-k sampling parameter.
        frequency_penalty: Penalty for repeating tokens (OpenAI).
        presence_penalty: Penalty for new tokens (OpenAI).
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
    thinking_enabled: bool = False
    thinking_budget: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

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
            "thinking_enabled": self.thinking_enabled,
            "thinking_budget": self.thinking_budget,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
