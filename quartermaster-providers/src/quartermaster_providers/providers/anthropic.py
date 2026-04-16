"""Anthropic LLM provider implementation (Claude family).

Implements AbstractLLMProvider for Anthropic's Claude models, supporting
text generation, streaming, tool calling, extended thinking, and vision.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, NoReturn

from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.config import LLMConfig
from quartermaster_providers.exceptions import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)
from quartermaster_providers.types import (
    NativeResponse,
    StructuredResponse,
    ThinkingResponse,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)

ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-3-7-sonnet-20250219": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}

DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(AbstractLLMProvider):
    """Anthropic LLM provider for Claude models.

    Supports text generation, tool calling, extended thinking, vision,
    and structured output.

    Args:
        api_key: Anthropic API key.
        base_url: Optional custom API endpoint.
    """

    PROVIDER_NAME = "anthropic"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic package required for AnthropicProvider. "
                "Install with: pip install quartermaster-providers[anthropic]"
            ) from e

        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """Translate Anthropic SDK exceptions to quartermaster-providers exceptions."""
        import anthropic

        if isinstance(e, anthropic.AuthenticationError):
            raise AuthenticationError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, anthropic.RateLimitError):
            raise RateLimitError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, anthropic.BadRequestError):
            msg = str(e).lower()
            if "context" in msg or "too many tokens" in msg or "too long" in msg:
                raise ContextLengthError(str(e), provider=self.PROVIDER_NAME) from e
            if "content" in msg and "filter" in msg:
                raise ContentFilterError(str(e), provider=self.PROVIDER_NAME) from e
            raise InvalidRequestError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, anthropic.APIStatusError):
            if e.status_code == 503 or e.status_code == 529:
                raise ServiceUnavailableError(str(e), provider=self.PROVIDER_NAME) from e
            raise ProviderError(
                str(e), provider=self.PROVIDER_NAME, status_code=e.status_code
            ) from e
        if isinstance(e, anthropic.APIConnectionError):
            raise ServiceUnavailableError(str(e), provider=self.PROVIDER_NAME) from e
        raise ProviderError(str(e), provider=self.PROVIDER_NAME) from e

    def _build_user_content(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> str | list[dict[str, Any]]:
        """Build the user-turn ``content`` field for the Anthropic API.

        Plain text requests use the string shortcut (``content: "..."``)
        so we don't bloat payloads for the common case. When the caller
        attached images via ``LLMConfig.images`` we switch to the
        structured list form and emit an ``image`` block per attachment
        before the text prompt — the order Anthropic documents for
        vision tool use.
        """
        if not config.images:
            return prompt
        blocks: list[dict[str, Any]] = []
        for b64_data, mime_type in config.images:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type or "image/jpeg",
                        "data": b64_data,
                    },
                }
            )
        blocks.append({"type": "text", "text": prompt})
        return blocks

    def _build_params(
        self,
        prompt: str,
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build Anthropic API request parameters."""
        params: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": self._build_user_content(prompt, config)}],
            "max_tokens": config.max_output_tokens or DEFAULT_MAX_TOKENS,
        }

        if config.system_message:
            params["system"] = config.system_message

        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.top_k is not None:
            params["top_k"] = config.top_k

        if tools:
            params["tools"] = tools

        if config.thinking_enabled and config.thinking_budget:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget,
            }
            # Anthropic requires temperature=1 when thinking is enabled
            params["temperature"] = 1.0

        # v0.4.0: thread timeouts through to the anthropic SDK via its
        # per-request ``timeout=`` kwarg. Accepts a scalar float or an
        # ``httpx.Timeout`` — ``_resolve_httpx_timeout`` picks the right
        # shape based on which of connect_timeout / read_timeout is set.
        timeout = self._resolve_httpx_timeout(config)
        if timeout is not None:
            params["timeout"] = timeout

        return params

    def _parse_usage(self, usage: Any) -> TokenUsage:
        """Parse Anthropic usage object to TokenUsage."""
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
        )

    async def list_models(self) -> list[str]:
        return list(ANTHROPIC_MODELS)

    def estimate_token_count(self, text: str, model: str) -> int:
        try:
            client = self._get_client()
            if hasattr(client, "count_tokens"):
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return int(len(text.split()) * 1.3)
                return int(loop.run_until_complete(client.count_tokens(text)))
        except Exception:
            pass
        return int(len(text.split()) * 1.3)

    def prepare_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {}),
        }

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        client = self._get_client()
        params = self._build_params(prompt, config)

        try:
            if config.stream:
                return self._stream_text(client, params)
            else:
                response = await client.messages.create(**params)
                text_parts = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)

                return TokenResponse(
                    content="".join(text_parts),
                    stop_reason=response.stop_reason,
                )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def _stream_text(
        self, client: Any, params: dict[str, Any]
    ) -> AsyncIterator[TokenResponse]:
        try:
            async with client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield TokenResponse(content=text)
                response = await stream.get_final_message()
                yield TokenResponse(
                    content="",
                    stop_reason=response.stop_reason,
                )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def generate_tool_parameters(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        config: LLMConfig,
    ) -> ToolCallResponse:
        client = self._get_client()
        prepared_tools = [self.prepare_tool(t) for t in tools]
        params = self._build_params(prompt, config, tools=prepared_tools)

        try:
            response = await client.messages.create(**params)

            text_parts = []
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            tool_name=block.name,
                            tool_id=block.id,
                            parameters=block.input if isinstance(block.input, dict) else {},
                        )
                    )

            return ToolCallResponse(
                text_content="".join(text_parts),
                tool_calls=tool_calls,
                stop_reason=response.stop_reason,
                usage=self._parse_usage(response.usage),
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def generate_native_response(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        config: LLMConfig | None = None,
    ) -> NativeResponse:
        if config is None:
            raise InvalidRequestError("config is required", provider=self.PROVIDER_NAME)

        client = self._get_client()
        prepared_tools = [self.prepare_tool(t) for t in tools] if tools else None
        params = self._build_params(prompt, config, tools=prepared_tools)

        try:
            response = await client.messages.create(**params)

            text_parts = []
            thinking_blocks = []
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking":
                    thinking_blocks.append(
                        ThinkingResponse(
                            thinking=block.thinking,
                            type="thinking",
                        )
                    )
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            tool_name=block.name,
                            tool_id=block.id,
                            parameters=block.input if isinstance(block.input, dict) else {},
                        )
                    )

            return NativeResponse(
                text_content="".join(text_parts),
                thinking=thinking_blocks,
                tool_calls=tool_calls,
                stop_reason=response.stop_reason,
                usage=self._parse_usage(response.usage),
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def generate_structured_response(
        self,
        prompt: str,
        response_schema: dict[str, Any] | type,
        config: LLMConfig,
    ) -> StructuredResponse:
        client = self._get_client()

        schema_dict: dict[str, Any]
        if isinstance(response_schema, type):
            if hasattr(response_schema, "__annotations__"):
                schema_dict = {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in response_schema.__annotations__},
                }
            else:
                schema_dict = {"type": "object"}
        else:
            schema_dict = response_schema

        json_prompt = (
            f"{prompt}\n\nRespond with valid JSON and nothing else. "
            f"The JSON must match this schema: {json.dumps(schema_dict)}"
        )
        params = self._build_params(json_prompt, config)

        try:
            response = await client.messages.create(**params)

            raw_output = ""
            for block in response.content:
                if block.type == "text":
                    raw_output += block.text

            try:
                structured = json.loads(raw_output)
            except json.JSONDecodeError:
                structured = {"raw": raw_output}

            return StructuredResponse(
                structured_output=structured,
                raw_output=raw_output,
                stop_reason=response.stop_reason,
                usage=self._parse_usage(response.usage),
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def transcribe(self, audio_path: str) -> str:
        raise ProviderError(
            "Anthropic does not support audio transcription. Use OpenAIProvider for transcription.",
            provider=self.PROVIDER_NAME,
        )

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        pricing = ANTHROPIC_PRICING.get(model)
        return pricing["input"] if pricing else None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        pricing = ANTHROPIC_PRICING.get(model)
        return pricing["output"] if pricing else None
