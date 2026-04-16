"""OpenAI LLM provider implementation (GPT-4o, GPT-4, o-series, Whisper).

Implements AbstractLLMProvider for OpenAI's API, supporting text generation,
streaming, tool calling, structured output, vision, and audio transcription.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, NoReturn, cast

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
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)

IMAGE_TOKEN_ESTIMATE = 1000

# Known OpenAI models
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
]

# Pricing per 1K tokens (USD) as of 2025
OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
}


def _is_o_series(model: str) -> bool:
    """Check if a model is an o-series reasoning model."""
    return model.startswith("o1") or model.startswith("o3")


def _extract_reasoning_text(obj: Any) -> str:
    """Pull reasoning text out of a non-standard OpenAI-compatible response.

    Some OpenAI-compatible servers (notably Ollama proxies for reasoning
    models like ``gemma4:26b``) leave ``content`` empty and put the user-
    visible answer in a ``reasoning`` or ``reasoning_content`` field.  This
    helper checks both attribute access (Pydantic model) and dict access so
    it works for any shape the SDK returns.

    The fallback is intentionally narrow: only non-empty string values
    trigger it, so a vanilla OpenAI response with empty content + empty
    reasoning still returns ``""`` (preserving "model said nothing"
    semantics for content-filtered or zero-temperature edge cases).
    """
    if obj is None:
        return ""
    for key in ("reasoning_content", "reasoning"):
        value = getattr(obj, key, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class OpenAIProvider(AbstractLLMProvider):
    """OpenAI LLM provider for GPT-4o, GPT-4, o-series, and Whisper.

    Args:
        api_key: OpenAI API key.
        organization_id: Optional OpenAI organization ID.
        base_url: Optional custom API endpoint (for Azure, proxies, etc.).
    """

    PROVIDER_NAME = "openai"

    def __init__(
        self,
        api_key: str,
        organization_id: str | None = None,
        base_url: str | None = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai package required for OpenAIProvider. "
                "Install with: pip install quartermaster-providers[openai]"
            ) from e

        self.api_key = api_key
        self.organization_id = organization_id
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization_id,
                base_url=self.base_url,
            )
        return self._client

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """Translate OpenAI SDK exceptions to quartermaster-providers exceptions."""
        import openai

        if isinstance(e, openai.AuthenticationError):
            raise AuthenticationError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, openai.RateLimitError):
            raise RateLimitError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, openai.BadRequestError):
            msg = str(e).lower()
            if "context_length" in msg or "maximum context" in msg or "too many tokens" in msg:
                raise ContextLengthError(str(e), provider=self.PROVIDER_NAME) from e
            if "content_filter" in msg or "content_policy" in msg:
                raise ContentFilterError(str(e), provider=self.PROVIDER_NAME) from e
            raise InvalidRequestError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, openai.APIStatusError):
            if e.status_code == 503:
                raise ServiceUnavailableError(str(e), provider=self.PROVIDER_NAME) from e
            raise ProviderError(
                str(e), provider=self.PROVIDER_NAME, status_code=e.status_code
            ) from e
        if isinstance(e, openai.APIConnectionError):
            raise ServiceUnavailableError(str(e), provider=self.PROVIDER_NAME) from e
        raise ProviderError(str(e), provider=self.PROVIDER_NAME) from e

    def _build_user_content(self, prompt: str, config: LLMConfig) -> str | list[dict[str, Any]]:
        """Build the user-turn ``content`` field for the Chat Completions API.

        Text-only requests keep the plain string shortcut so tokens /
        payloads stay unchanged for 99% of callsites. When the caller
        attached images via ``LLMConfig.images`` we emit the structured
        content-part list the OpenAI SDK expects: each image becomes an
        ``image_url`` part with a ``data:<mime>;base64,<data>`` URL, and
        the original text prompt comes last.
        """
        if not config.images:
            return prompt
        parts: list[dict[str, Any]] = []
        for b64_data, mime_type in config.images:
            # Wrap as a data URI — cheapest path that works with every
            # OpenAI-compatible server (including Ollama's /v1 proxy).
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type or 'image/jpeg'};base64,{b64_data}",
                    },
                }
            )
        parts.append({"type": "text", "text": prompt})
        return parts

    def _build_messages(self, prompt: str, config: LLMConfig) -> list[dict[str, Any]]:
        """Build OpenAI messages array from prompt and config."""
        messages: list[dict[str, Any]] = []
        if config.system_message and not _is_o_series(config.model):
            messages.append({"role": "system", "content": config.system_message})
        messages.append({"role": "user", "content": self._build_user_content(prompt, config)})
        return messages

    def _build_params(
        self,
        config: LLMConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build request parameters from config."""
        params: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
        }

        if not _is_o_series(config.model):
            params["temperature"] = config.temperature
        if config.max_output_tokens:
            params["max_tokens"] = config.max_output_tokens
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            params["presence_penalty"] = config.presence_penalty
        if tools:
            params["tools"] = tools
        if response_format:
            params["response_format"] = response_format
        if config.stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}

        # v0.4.0: thread timeouts through to the openai SDK. The SDK
        # accepts ``timeout=`` on every request; passing ``httpx.Timeout``
        # is honoured through its underlying httpx transport. Leave
        # unset when neither connect_timeout nor read_timeout is
        # configured so the SDK default keeps applying.
        timeout = self._resolve_httpx_timeout(config)
        if timeout is not None:
            params["timeout"] = timeout

        return params

    async def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            models = await client.models.list()
            return sorted([m.id for m in models.data])
        except Exception:
            return list(OPENAI_MODELS)

    def estimate_token_count(self, text: str, model: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)

    def prepare_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        client = self._get_client()
        messages = self._build_messages(prompt, config)
        params = self._build_params(config, messages)

        try:
            if config.stream:
                return self._stream_text(client, params)
            else:
                response = await client.chat.completions.create(**params)
                choice = response.choices[0]
                content = choice.message.content or ""
                if not content:
                    # Some OpenAI-compatible reasoning models leave content
                    # empty and place the answer in a ``reasoning`` field.
                    content = _extract_reasoning_text(choice.message)
                return TokenResponse(
                    content=content,
                    stop_reason=choice.finish_reason,
                )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def _stream_text(
        self, client: Any, params: dict[str, Any]
    ) -> AsyncIterator[TokenResponse]:
        try:
            stream = await client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason
                    if delta and delta.content:
                        yield TokenResponse(
                            content=delta.content,
                            stop_reason=finish_reason,
                        )
                    else:
                        # OpenAI-compatible reasoning models stream their
                        # answer through ``reasoning_content`` (or ``reasoning``)
                        # when ``content`` is absent — surface that as visible
                        # text so callers don't see an empty stream.
                        reasoning_chunk = _extract_reasoning_text(delta) if delta else ""
                        if reasoning_chunk:
                            yield TokenResponse(
                                content=reasoning_chunk,
                                stop_reason=finish_reason,
                            )
                        elif finish_reason:
                            yield TokenResponse(
                                content="",
                                stop_reason=finish_reason,
                            )
                if hasattr(chunk, "usage") and chunk.usage:
                    yield TokenResponse(
                        content="",
                        stop_reason="usage",
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
        messages = self._build_messages(prompt, config)
        prepared_tools = [self.prepare_tool(t) for t in tools]
        params = self._build_params(config, messages, tools=prepared_tools)
        params.pop("stream", None)
        params.pop("stream_options", None)

        try:
            response = await client.chat.completions.create(**params)
            choice = response.choices[0]
            message = choice.message

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        parameters = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        parameters = {"raw": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            tool_id=tc.id,
                            parameters=parameters,
                        )
                    )

            usage = None
            if response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )

            return ToolCallResponse(
                text_content=message.content or "",
                tool_calls=tool_calls,
                stop_reason=choice.finish_reason,
                usage=usage,
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
        messages = self._build_messages(prompt, config)
        prepared_tools = [self.prepare_tool(t) for t in tools] if tools else None
        params = self._build_params(config, messages, tools=prepared_tools)
        params.pop("stream", None)
        params.pop("stream_options", None)

        try:
            response = await client.chat.completions.create(**params)
            choice = response.choices[0]
            message = choice.message

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        parameters = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        parameters = {"raw": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            tool_id=tc.id,
                            parameters=parameters,
                        )
                    )

            usage = None
            if response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )

            text_content = message.content or ""
            if not text_content:
                text_content = _extract_reasoning_text(message)
            return NativeResponse(
                text_content=text_content,
                thinking=[],
                tool_calls=tool_calls,
                stop_reason=choice.finish_reason,
                usage=usage,
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
        messages = self._build_messages(prompt, config)

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
            f"{prompt}\n\nRespond with valid JSON matching this schema: {json.dumps(schema_dict)}"
        )
        messages[-1]["content"] = json_prompt

        response_format = {"type": "json_object"}
        params = self._build_params(config, messages, response_format=response_format)
        params.pop("stream", None)
        params.pop("stream_options", None)

        try:
            response = await client.chat.completions.create(**params)
            choice = response.choices[0]
            raw_output = choice.message.content or ""

            try:
                structured = json.loads(raw_output)
            except json.JSONDecodeError:
                structured = {"raw": raw_output}

            usage = None
            if response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )

            return StructuredResponse(
                structured_output=structured,
                raw_output=raw_output,
                stop_reason=choice.finish_reason,
                usage=usage,
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        client = self._get_client()
        try:
            with open(audio_path, "rb") as audio_file:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            return cast(str, response.text)
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        pricing = OPENAI_PRICING.get(model)
        return pricing["input"] if pricing else None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        pricing = OPENAI_PRICING.get(model)
        return pricing["output"] if pricing else None
