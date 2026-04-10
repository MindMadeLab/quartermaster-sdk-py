"""Tests for provider implementations.

Tests provider initialization, tool formatting, token estimation,
pricing, and error handling. Does NOT make real API calls.
"""

import pytest

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.exceptions import ProviderError
from quartermaster_providers.types import ToolDefinition


# ── OpenAI Provider ──────────────────────────────────────────────────────


class TestOpenAIProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from quartermaster_providers.providers.openai import OpenAIProvider

        self.provider = OpenAIProvider(api_key="sk-test-fake")

    def test_init(self):
        assert self.provider.api_key == "sk-test-fake"
        assert self.provider.organization_id is None
        assert self.provider.base_url is None
        assert self.provider._client is None

    def test_init_with_options(self):
        from quartermaster_providers.providers.openai import OpenAIProvider

        p = OpenAIProvider(
            api_key="sk-test",
            organization_id="org-123",
            base_url="https://custom.api/v1",
        )
        assert p.organization_id == "org-123"
        assert p.base_url == "https://custom.api/v1"

    def test_provider_name(self):
        from quartermaster_providers.providers.openai import OpenAIProvider

        assert OpenAIProvider.PROVIDER_NAME == "openai"

    def test_prepare_tool(self):
        tool: ToolDefinition = {
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        result = self.provider.prepare_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search the web"
        assert result["function"]["parameters"]["type"] == "object"

    def test_prepare_tool_empty(self):
        tool: ToolDefinition = {"name": "", "description": "", "input_schema": {}}
        result = self.provider.prepare_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == ""

    def test_estimate_token_count(self):
        count = self.provider.estimate_token_count("hello world test", "gpt-4o")
        assert count > 0

    def test_pricing_known_model(self):
        assert self.provider.get_cost_per_1k_input_tokens("gpt-4o") is not None
        assert self.provider.get_cost_per_1k_output_tokens("gpt-4o") is not None

    def test_pricing_unknown_model(self):
        assert self.provider.get_cost_per_1k_input_tokens("unknown") is None
        assert self.provider.get_cost_per_1k_output_tokens("unknown") is None

    def test_estimate_cost(self):
        cost = self.provider.estimate_cost("hello world", "gpt-4o", output_tokens=100)
        assert cost is not None
        assert cost > 0

    def test_estimate_cost_unknown_model(self):
        cost = self.provider.estimate_cost("hello", "unknown-model")
        assert cost is None

    def test_build_messages_with_system(self):
        config = LLMConfig(model="gpt-4o", provider="openai", system_message="Be helpful")
        messages = self.provider._build_messages("Hello", config)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"

    def test_build_messages_no_system(self):
        config = LLMConfig(model="gpt-4o", provider="openai")
        messages = self.provider._build_messages("Hello", config)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_build_messages_o_series_no_system(self):
        config = LLMConfig(model="o1", provider="openai", system_message="Be helpful")
        messages = self.provider._build_messages("Hello", config)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_build_params_basic(self):
        config = LLMConfig(
            model="gpt-4o", provider="openai", temperature=0.5, max_output_tokens=1000
        )
        messages = [{"role": "user", "content": "Hi"}]
        params = self.provider._build_params(config, messages)
        assert params["model"] == "gpt-4o"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 1000

    def test_build_params_streaming(self):
        config = LLMConfig(model="gpt-4o", provider="openai", stream=True)
        messages = [{"role": "user", "content": "Hi"}]
        params = self.provider._build_params(config, messages)
        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": True}

    def test_build_params_o_series_no_temperature(self):
        config = LLMConfig(model="o1", provider="openai", temperature=0.5)
        messages = [{"role": "user", "content": "Hi"}]
        params = self.provider._build_params(config, messages)
        assert "temperature" not in params

    def test_build_params_with_tools(self):
        config = LLMConfig(model="gpt-4o", provider="openai")
        messages = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "fn"}}]
        params = self.provider._build_params(config, messages, tools=tools)
        assert params["tools"] == tools

    def test_build_params_optional_fields(self):
        config = LLMConfig(
            model="gpt-4o",
            provider="openai",
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        messages = [{"role": "user", "content": "Hi"}]
        params = self.provider._build_params(config, messages)
        assert params["top_p"] == 0.9
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.3

    def test_lazy_client_creation(self):
        assert self.provider._client is None
        client = self.provider._get_client()
        assert client is not None
        # Same client returned
        assert self.provider._get_client() is client


# ── Anthropic Provider ───────────────────────────────────────────────────


class TestAnthropicProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from quartermaster_providers.providers.anthropic import AnthropicProvider

        self.provider = AnthropicProvider(api_key="sk-ant-test-fake")

    def test_init(self):
        assert self.provider.api_key == "sk-ant-test-fake"
        assert self.provider.base_url is None
        assert self.provider._client is None

    def test_provider_name(self):
        from quartermaster_providers.providers.anthropic import AnthropicProvider

        assert AnthropicProvider.PROVIDER_NAME == "anthropic"

    def test_prepare_tool(self):
        tool: ToolDefinition = {
            "name": "weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
        result = self.provider.prepare_tool(tool)
        assert result["name"] == "weather"
        assert result["description"] == "Get weather"
        assert "input_schema" in result

    def test_pricing_known_model(self):
        assert self.provider.get_cost_per_1k_input_tokens("claude-sonnet-4-20250514") is not None
        assert self.provider.get_cost_per_1k_output_tokens("claude-3-opus-20240229") is not None

    def test_pricing_unknown_model(self):
        assert self.provider.get_cost_per_1k_input_tokens("unknown") is None

    def test_build_params_basic(self):
        config = LLMConfig(
            model="claude-3-opus-20240229",
            provider="anthropic",
            system_message="You are helpful.",
            max_output_tokens=2000,
        )
        params = self.provider._build_params("Hello", config)
        assert params["model"] == "claude-3-opus-20240229"
        assert params["system"] == "You are helpful."
        assert params["max_tokens"] == 2000
        assert params["messages"] == [{"role": "user", "content": "Hello"}]

    def test_build_params_thinking(self):
        config = LLMConfig(
            model="claude-3-7-sonnet-20250219",
            provider="anthropic",
            thinking_enabled=True,
            thinking_budget=10000,
        )
        params = self.provider._build_params("Hello", config)
        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 10000
        assert params["temperature"] == 1.0

    def test_build_params_with_tools(self):
        config = LLMConfig(
            model="claude-3-opus-20240229",
            provider="anthropic",
        )
        tools = [{"name": "fn", "input_schema": {}}]
        params = self.provider._build_params("Hello", config, tools=tools)
        assert params["tools"] == tools

    def test_build_params_default_max_tokens(self):
        config = LLMConfig(model="claude-3-opus-20240229", provider="anthropic")
        params = self.provider._build_params("Hello", config)
        assert params["max_tokens"] == 4096  # DEFAULT_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_transcribe_raises(self):
        with pytest.raises(ProviderError, match="does not support"):
            await self.provider.transcribe("/path/to/audio.wav")

    @pytest.mark.asyncio
    async def test_list_models(self):
        models = await self.provider.list_models()
        assert len(models) > 0
        assert any("claude" in m for m in models)

    def test_estimate_token_count(self):
        count = self.provider.estimate_token_count("hello world test", "claude-3-opus")
        assert count > 0

    def test_lazy_client(self):
        assert self.provider._client is None
        client = self.provider._get_client()
        assert client is not None
        assert self.provider._get_client() is client


# ── Google Provider ──────────────────────────────────────────────────────


class TestGoogleProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("google.generativeai")
        from quartermaster_providers.providers.google import GoogleProvider

        self.provider = GoogleProvider(api_key="fake-google-key")

    def test_init(self):
        assert self.provider.api_key == "fake-google-key"
        assert self.provider._configured is False

    def test_provider_name(self):
        from quartermaster_providers.providers.google import GoogleProvider

        assert GoogleProvider.PROVIDER_NAME == "google"

    def test_prepare_tool(self):
        tool: ToolDefinition = {
            "name": "calc",
            "description": "Calculate",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
        result = self.provider.prepare_tool(tool)
        assert result["name"] == "calc"
        assert "parameters" in result

    def test_clean_schema_for_google(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "A name"}},
            "required": ["name"],
            "additionalProperties": False,
            "$schema": "http://json-schema.org",
        }
        cleaned = self.provider._clean_schema_for_google(schema)
        assert "type" in cleaned
        assert "properties" in cleaned
        assert "required" in cleaned
        assert "additionalProperties" not in cleaned
        assert "$schema" not in cleaned

    def test_clean_schema_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "additionalProperties": True,
                    },
                },
            },
        }
        cleaned = self.provider._clean_schema_for_google(schema)
        items_prop = cleaned["properties"]["items"]
        assert "additionalProperties" not in items_prop.get("items", {})

    def test_pricing_known_model(self):
        assert self.provider.get_cost_per_1k_input_tokens("gemini-2.0-flash") is not None
        assert self.provider.get_cost_per_1k_output_tokens("gemini-1.5-pro") is not None

    def test_pricing_unknown_model(self):
        assert self.provider.get_cost_per_1k_input_tokens("unknown") is None

    @pytest.mark.asyncio
    async def test_transcribe_raises(self):
        with pytest.raises(ProviderError, match="not yet implemented"):
            await self.provider.transcribe("/path/to/audio.wav")

    def test_estimate_token_count(self):
        count = self.provider.estimate_token_count("hello world test", "gemini-1.5-pro")
        assert count > 0


# ── Groq Provider ────────────────────────────────────────────────────────


class TestGroqProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from quartermaster_providers.providers.groq import GroqProvider

        self.provider = GroqProvider(api_key="gsk-test-fake")

    def test_init(self):
        from quartermaster_providers.providers.groq import GROQ_BASE_URL

        assert self.provider.api_key == "gsk-test-fake"
        assert self.provider.base_url == GROQ_BASE_URL

    def test_provider_name(self):
        from quartermaster_providers.providers.groq import GroqProvider

        assert GroqProvider.PROVIDER_NAME == "groq"

    def test_pricing(self):
        assert self.provider.get_cost_per_1k_input_tokens("llama-3.3-70b-versatile") is not None
        assert self.provider.get_cost_per_1k_output_tokens("mixtral-8x7b-32768") is not None

    def test_pricing_unknown(self):
        assert self.provider.get_cost_per_1k_input_tokens("unknown") is None

    def test_estimate_token_count(self):
        count = self.provider.estimate_token_count("hello world", "llama-3.3-70b")
        assert count > 0

    def test_inherits_prepare_tool(self):
        tool: ToolDefinition = {
            "name": "fn",
            "description": "desc",
            "input_schema": {"type": "object"},
        }
        result = self.provider.prepare_tool(tool)
        assert result["type"] == "function"


# ── xAI Provider ─────────────────────────────────────────────────────────


class TestXAIProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from quartermaster_providers.providers.xai import XAIProvider

        self.provider = XAIProvider(api_key="xai-test-fake")

    def test_init(self):
        from quartermaster_providers.providers.xai import XAI_BASE_URL

        assert self.provider.api_key == "xai-test-fake"
        assert self.provider.base_url == XAI_BASE_URL

    def test_provider_name(self):
        from quartermaster_providers.providers.xai import XAIProvider

        assert XAIProvider.PROVIDER_NAME == "xai"

    @pytest.mark.asyncio
    async def test_transcribe_raises(self):
        with pytest.raises(ProviderError, match="does not support"):
            await self.provider.transcribe("/path/to/audio.wav")

    def test_pricing(self):
        assert self.provider.get_cost_per_1k_input_tokens("grok-2") is not None
        assert self.provider.get_cost_per_1k_output_tokens("grok-2-mini") is not None

    def test_pricing_unknown(self):
        assert self.provider.get_cost_per_1k_input_tokens("unknown") is None


# ── OpenAI Compatible Provider ───────────────────────────────────────────


class TestOpenAICompatibleProvider:
    def test_init_bearer(self):
        from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        assert p.base_url == "http://localhost:11434/v1"
        assert p.api_key == "ollama"

    def test_init_no_auth(self):
        from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1",
            auth_method="none",
        )
        assert p.api_key == "no-key"

    def test_custom_provider_name(self):
        from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            provider_name="ollama",
        )
        assert p.PROVIDER_NAME == "ollama"

    def test_estimate_token_count(self):
        from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(base_url="http://localhost/v1")
        count = p.estimate_token_count("hello world test", "custom-model")
        assert count > 0

    def test_no_pricing(self):
        from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(base_url="http://localhost/v1")
        assert p.get_cost_per_1k_input_tokens("any") is None
        assert p.get_cost_per_1k_output_tokens("any") is None

    def test_inherits_openai_tools(self):
        from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(base_url="http://localhost/v1")
        tool: ToolDefinition = {
            "name": "fn",
            "description": "desc",
            "input_schema": {"type": "object"},
        }
        result = p.prepare_tool(tool)
        assert result["type"] == "function"
