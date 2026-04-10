"""Abstract base class for LLM provider implementations.

All LLM provider implementations (OpenAI, Anthropic, Google, etc.) inherit
from AbstractLLMProvider and implement its abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, TypeVar

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.types import (
    NativeResponse,
    StructuredResponse,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
)

T = TypeVar("T")


class AbstractLLMProvider(ABC):
    """Base class for all LLM provider implementations.

    Providers must implement 8 abstract methods to support different
    response types and capabilities.

    Example:
        class MyProvider(AbstractLLMProvider):
            async def list_models(self) -> list[str]:
                return ["model-1", "model-2"]

            async def generate_text_response(
                self,
                prompt: str,
                config: LLMConfig,
            ) -> TokenResponse | AsyncIterator[TokenResponse]:
                # Implementation here
                ...
    """

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models for this provider.

        Returns:
            List of model identifiers (e.g., 'gpt-4', 'claude-3-opus').
        """
        ...

    @abstractmethod
    def estimate_token_count(self, text: str, model: str) -> int:
        """Estimate token count for text without calling the API.

        Args:
            text: Text to estimate tokens for.
            model: Model identifier.

        Returns:
            Estimated token count.
        """
        ...

    @abstractmethod
    def prepare_tool(self, tool: ToolDefinition) -> Any:
        """Transform a tool definition to provider-specific format.

        Different providers have different tool/function calling formats.
        This method translates from the unified ToolDefinition to the
        provider's native format.

        Args:
            tool: Unified tool definition.

        Returns:
            Provider-specific tool definition.
        """
        ...

    @abstractmethod
    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        """Generate a text response from a prompt.

        Args:
            prompt: Input prompt text.
            config: LLM configuration (temperature, max_tokens, etc.).

        Returns:
            Single TokenResponse if not streaming, otherwise AsyncIterator
            of TokenResponse chunks. Check config.stream to determine which.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limited.
            ProviderError: For other provider-specific errors.
        """
        ...

    @abstractmethod
    async def generate_tool_parameters(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        config: LLMConfig,
    ) -> ToolCallResponse:
        """Generate tool calls with parameters.

        The model receives the list of available tools and generates
        function calls it thinks should be made.

        Args:
            prompt: Input prompt requesting tool use.
            tools: List of available tools the model can call.
            config: LLM configuration.

        Returns:
            ToolCallResponse with tool_calls and any text_content.

        Raises:
            ProviderError: If provider doesn't support tool calling.
        """
        ...

    @abstractmethod
    async def generate_native_response(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        config: LLMConfig | None = None,
    ) -> NativeResponse:
        """Generate the complete/native response from the model.

        This returns all possible response types in one call: text,
        thinking blocks, tool calls, etc.

        Args:
            prompt: Input prompt.
            tools: Optional list of available tools.
            config: LLM configuration.

        Returns:
            NativeResponse containing text, thinking, tool calls, usage.
        """
        ...

    @abstractmethod
    async def generate_structured_response(
        self,
        prompt: str,
        response_schema: dict[str, Any] | type,
        config: LLMConfig,
    ) -> StructuredResponse:
        """Generate output conforming to a JSON schema.

        Some models support guided generation that guarantees the output
        conforms to a provided JSON schema. This is useful for extracting
        structured data.

        Args:
            prompt: Input prompt requesting structured output.
            response_schema: JSON Schema or Pydantic model class.
            config: LLM configuration.

        Returns:
            StructuredResponse with parsed structured_output.

        Raises:
            ProviderError: If provider doesn't support structured output.
        """
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
    ) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, m4a, flac, etc.).

        Returns:
            Transcribed text.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ProviderError: If provider doesn't support transcription.
        """
        ...

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        """Get the cost per 1K input tokens for a model.

        Args:
            model: Model identifier.

        Returns:
            Cost in USD per 1K input tokens, or None if unknown.
        """
        return None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        """Get the cost per 1K output tokens for a model.

        Args:
            model: Model identifier.

        Returns:
            Cost in USD per 1K output tokens, or None if unknown.
        """
        return None

    def estimate_cost(
        self,
        text: str,
        model: str,
        output_tokens: int | None = None,
    ) -> float | None:
        """Estimate cost for a request.

        Args:
            text: Input text to estimate.
            model: Model identifier.
            output_tokens: Estimated output tokens (if known).

        Returns:
            Estimated cost in USD, or None if pricing unavailable.
        """
        input_cost_per_1k = self.get_cost_per_1k_input_tokens(model)
        if input_cost_per_1k is None:
            return None

        input_tokens = self.estimate_token_count(text, model)
        input_cost = (input_tokens / 1000) * input_cost_per_1k

        if output_tokens is None:
            return input_cost

        output_cost_per_1k = self.get_cost_per_1k_output_tokens(model)
        if output_cost_per_1k is None:
            return input_cost

        output_cost = (output_tokens / 1000) * output_cost_per_1k
        return input_cost + output_cost
