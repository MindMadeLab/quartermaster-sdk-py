"""Google LLM provider implementation (Gemini).

Implements AbstractLLMProvider for Google's Gemini models, supporting
text generation, streaming, tool calling, structured output, and vision.
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
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)

GOOGLE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

GOOGLE_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
}


class GoogleProvider(AbstractLLMProvider):
    """Google LLM provider for Gemini models.

    Supports text generation, tool calling, vision, and structured output.

    Args:
        api_key: Google API key.
    """

    PROVIDER_NAME = "google"

    def __init__(self, api_key: str):
        try:
            import google.generativeai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "google-generativeai package required for GoogleProvider. "
                "Install with: pip install quartermaster-providers[google]"
            ) from e

        self.api_key = api_key
        self._configured = False

    def _ensure_configured(self):
        if not self._configured:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._configured = True

    def _get_model(self, config: LLMConfig):
        import google.generativeai as genai

        self._ensure_configured()

        generation_config: dict[str, Any] = {}
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature
        if config.max_output_tokens:
            generation_config["max_output_tokens"] = config.max_output_tokens
        if config.top_p is not None:
            generation_config["top_p"] = config.top_p
        if config.top_k is not None:
            generation_config["top_k"] = config.top_k

        system_instruction = config.system_message if config.system_message else None

        return genai.GenerativeModel(
            model_name=config.model,
            generation_config=generation_config if generation_config else None,  # type: ignore[arg-type]
            system_instruction=system_instruction,
        )

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """Translate Google SDK exceptions to quartermaster-providers exceptions."""
        msg = str(e).lower()
        if "api_key" in msg or "authentication" in msg or "permission" in msg:
            raise AuthenticationError(str(e), provider=self.PROVIDER_NAME) from e
        if "quota" in msg or "rate" in msg or "resource_exhausted" in msg:
            raise RateLimitError(str(e), provider=self.PROVIDER_NAME) from e
        if "safety" in msg or "blocked" in msg or "harm" in msg:
            raise ContentFilterError(str(e), provider=self.PROVIDER_NAME) from e
        if "context" in msg or "too long" in msg or "token" in msg and "limit" in msg:
            raise ContextLengthError(str(e), provider=self.PROVIDER_NAME) from e
        if "not found" in msg or "invalid model" in msg:
            raise InvalidRequestError(str(e), provider=self.PROVIDER_NAME) from e
        if "unavailable" in msg or "internal" in msg:
            raise ServiceUnavailableError(str(e), provider=self.PROVIDER_NAME) from e
        raise ProviderError(str(e), provider=self.PROVIDER_NAME) from e

    async def list_models(self) -> list[str]:
        try:
            import google.generativeai as genai

            self._ensure_configured()
            models = []
            for m in genai.list_models():
                if "generateContent" in (m.supported_generation_methods or []):
                    models.append(m.name.replace("models/", ""))
            return sorted(models) if models else list(GOOGLE_MODELS)
        except Exception:
            return list(GOOGLE_MODELS)

    def estimate_token_count(self, text: str, model: str) -> int:
        try:
            import google.generativeai as genai

            self._ensure_configured()
            m = genai.GenerativeModel(model)
            result = m.count_tokens(text)
            return result.total_tokens
        except Exception:
            return int(len(text.split()) * 1.3)

    def prepare_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert to Google's function declaration format."""
        schema = tool.get("input_schema", {})
        # Google uses a slightly different schema format — strip unsupported keys
        clean_schema = self._clean_schema_for_google(schema)
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": clean_schema,
        }

    def _clean_schema_for_google(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Clean JSON Schema for Google's more restrictive format."""
        cleaned: dict[str, Any] = {}
        for key, value in schema.items():
            if key in ("type", "properties", "required", "description", "items", "enum"):
                if isinstance(value, dict):
                    cleaned[key] = self._clean_schema_for_google(value)
                elif isinstance(value, list):
                    cleaned[key] = [
                        self._clean_schema_for_google(v) if isinstance(v, dict) else v
                        for v in value
                    ]
                else:
                    cleaned[key] = value
        return cleaned

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        model = self._get_model(config)

        try:
            if config.stream:
                return self._stream_text(model, prompt)
            else:
                response = await model.generate_content_async(prompt)
                text = response.text if response.text else ""
                return TokenResponse(
                    content=text,
                    stop_reason=self._map_finish_reason(response),
                )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def _stream_text(self, model: Any, prompt: str) -> AsyncIterator[TokenResponse]:
        try:
            response = await model.generate_content_async(prompt, stream=True)
            async for chunk in response:
                if chunk.text:
                    yield TokenResponse(content=chunk.text)
            yield TokenResponse(content="", stop_reason="end_turn")
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
        import google.generativeai as genai

        model = self._get_model(config)
        prepared = [self.prepare_tool(t) for t in tools]

        google_tools = [genai.types.Tool(function_declarations=prepared)]

        try:
            response = await model.generate_content_async(
                prompt,
                tools=google_tools,
            )

            text_parts = []
            tool_calls = []

            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            tool_name=fc.name,
                            tool_id=f"google_{fc.name}",
                            parameters=dict(fc.args) if fc.args else {},
                        )
                    )

            usage = self._extract_usage(response)
            return ToolCallResponse(
                text_content="".join(text_parts),
                tool_calls=tool_calls,
                stop_reason=self._map_finish_reason(response),
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

        import google.generativeai as genai

        model = self._get_model(config)

        kwargs: dict[str, Any] = {}
        if tools:
            prepared = [self.prepare_tool(t) for t in tools]
            kwargs["tools"] = [genai.types.Tool(function_declarations=prepared)]

        try:
            response = await model.generate_content_async(prompt, **kwargs)

            text_parts = []
            tool_calls = []

            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            tool_name=fc.name,
                            tool_id=f"google_{fc.name}",
                            parameters=dict(fc.args) if fc.args else {},
                        )
                    )

            return NativeResponse(
                text_content="".join(text_parts),
                thinking=[],
                tool_calls=tool_calls,
                stop_reason=self._map_finish_reason(response),
                usage=self._extract_usage(response),
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
        model = self._get_model(config)

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

        try:
            response = await model.generate_content_async(json_prompt)
            raw_output = response.text if response.text else ""

            try:
                structured = json.loads(raw_output)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re

                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
                if json_match:
                    try:
                        structured = json.loads(json_match.group(1).strip())
                    except json.JSONDecodeError:
                        structured = {"raw": raw_output}
                else:
                    structured = {"raw": raw_output}

            return StructuredResponse(
                structured_output=structured,
                raw_output=raw_output,
                stop_reason=self._map_finish_reason(response),
                usage=self._extract_usage(response),
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def transcribe(self, audio_path: str) -> str:
        raise ProviderError(
            "Google transcription not yet implemented. Use OpenAIProvider for audio transcription.",
            provider=self.PROVIDER_NAME,
        )

    def _map_finish_reason(self, response: Any) -> str | None:
        """Map Google's finish reason to a standard string."""
        try:
            candidate = response.candidates[0]
            reason = candidate.finish_reason
            mapping = {1: "end_turn", 2: "max_tokens", 3: "safety", 4: "recitation", 5: "other"}
            return mapping.get(reason, str(reason) if reason else None)
        except (IndexError, AttributeError):
            return None

    def _extract_usage(self, response: Any) -> TokenUsage | None:
        """Extract token usage from Google response."""
        try:
            metadata = response.usage_metadata
            return TokenUsage(
                input_tokens=metadata.prompt_token_count or 0,
                output_tokens=metadata.candidates_token_count or 0,
            )
        except (AttributeError, TypeError):
            return None

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        pricing = GOOGLE_PRICING.get(model)
        return pricing["input"] if pricing else None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        pricing = GOOGLE_PRICING.get(model)
        return pricing["output"] if pricing else None
