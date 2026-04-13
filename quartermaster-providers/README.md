# quartermaster-providers

Unified multi-LLM provider abstraction for Python. Write once, run against OpenAI, Anthropic, Google, Groq, xAI, or any OpenAI-compatible endpoint.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-providers.svg)](https://pypi.org/project/quartermaster-providers/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **6 Providers**: OpenAI, Anthropic, Google, Groq, xAI, plus a generic OpenAI-compatible adapter
- **Streaming**: Async generators for token-by-token responses
- **Tool Calling**: Unified interface for function/tool invocation across all providers
- **Structured Output**: JSON-schema-constrained response generation
- **Extended Thinking**: Claude and o-series reasoning chains
- **Vision**: Image understanding on supported models
- **Transcription**: Audio-to-text via OpenAI Whisper
- **Token Counting**: Estimate tokens and cost before making requests
- **Provider Registry**: Register providers once, resolve by name or model pattern
- **Testing Utilities**: `MockProvider` and `InMemoryHistory` for unit tests
- **Type-Safe**: Dataclass responses with full type hints

## Installation

```bash
pip install quartermaster-providers
```

Install with provider-specific extras:

```bash
pip install quartermaster-providers[openai]
pip install quartermaster-providers[anthropic]
pip install quartermaster-providers[openai,anthropic,google]
pip install quartermaster-providers[all]
```

## Supported Providers

| Provider | Class | Models (examples) |
|----------|-------|-------------------|
| OpenAI | `OpenAIProvider` | gpt-4o, gpt-4-turbo, o1, o3-mini |
| Anthropic | `AnthropicProvider` | claude-sonnet-4-20250514, claude-3-haiku |
| Google | `GoogleProvider` | gemini-1.5-pro, gemini-pro |
| Groq | `GroqProvider` | llama-3-70b, mixtral-8x7b |
| xAI | `XAIProvider` | grok-2, grok-2-mini |
| Quartermaster | `QuartermasterProvider` | All models via one API key |
| Custom | `OpenAICompatibleProvider` | Any OpenAI-compatible API |

### Local / Self-Hosted Providers

| Provider | Class | Description |
|----------|-------|-------------|
| Ollama | `OllamaProvider` | Local models via Ollama |
| vLLM | `VLLMProvider` | High-throughput inference server |
| LM Studio | `LMStudioProvider` | Desktop LLM app |
| TGI | `TGIProvider` | HuggingFace Text Generation Inference |
| LocalAI | `LocalAIProvider` | OpenAI-compatible local server |
| llama.cpp | `LlamaCppProvider` | llama.cpp HTTP server |

Register local providers with one line:

```python
registry = ProviderRegistry()
registry.register_local("ollama")  # Auto-discovers models
```

## Quick Start

### Text Generation

```python
import asyncio
from quartermaster_providers import LLMConfig
from quartermaster_providers.providers import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key="sk-...")
    config = LLMConfig(
        model="gpt-4o",
        provider="openai",
        temperature=0.7,
        max_output_tokens=1024,
    )

    response = await provider.generate_text_response(
        prompt="Explain gradient descent in two sentences.",
        config=config,
    )
    print(response.content)  # str
    print(response.stop_reason)  # "end_turn", "max_tokens", etc.

asyncio.run(main())
```

### Tool Calling

```python
import asyncio
from quartermaster_providers import LLMConfig, ToolDefinition
from quartermaster_providers.providers import AnthropicProvider

async def main():
    provider = AnthropicProvider(api_key="sk-ant-...")
    config = LLMConfig(model="claude-sonnet-4-20250514", provider="anthropic")

    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get current weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        ),
    ]

    response = await provider.generate_tool_parameters(
        prompt="What is the weather in Tokyo?",
        tools=tools,
        config=config,
    )

    for call in response.tool_calls:
        print(f"{call.tool_name}({call.parameters})")
        # get_weather({'location': 'Tokyo'})

    print(f"Usage: {response.usage.total_tokens} tokens")

asyncio.run(main())
```

### Tool Calling with quartermaster-tools

Tools created with `@tool()` integrate directly via `ToolDescriptor`:

```python
from quartermaster_tools import tool

@tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: The city name to look up.
    """
    return {"city": city, "temperature": 22}

# Convert to provider-compatible format
tool_def = get_weather.info().to_anthropic_tools()
# Or for OpenAI:
tool_def = get_weather.info().to_openai_tools()
```

### Streaming

```python
import asyncio
from quartermaster_providers import LLMConfig
from quartermaster_providers.providers import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key="sk-...")
    config = LLMConfig(model="gpt-4o", provider="openai", stream=True)

    async for chunk in await provider.generate_text_response(
        prompt="Write a haiku about Python.",
        config=config,
    ):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### Structured Output

```python
import asyncio
from quartermaster_providers import LLMConfig
from quartermaster_providers.providers import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key="sk-...")
    config = LLMConfig(model="gpt-4o", provider="openai")

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "summary", "topics"],
    }

    response = await provider.generate_structured_response(
        prompt="Analyze the concept of reinforcement learning.",
        response_schema=schema,
        config=config,
    )

    print(response.structured_output["title"])
    print(response.structured_output["topics"])

asyncio.run(main())
```

## API Reference

### LLMConfig

Controls request behavior across all providers.

```python
from quartermaster_providers import LLMConfig

config = LLMConfig(
    model="gpt-4o",             # Provider model identifier
    provider="openai",          # Provider name
    stream=False,               # Stream token-by-token
    temperature=0.7,            # 0.0 (deterministic) to 2.0 (creative)
    system_message=None,        # System prompt
    max_input_tokens=None,      # Input token limit
    max_output_tokens=None,     # Output token limit
    max_messages=None,          # Conversation context limit
    vision=False,               # Enable image understanding
    thinking_enabled=False,     # Extended thinking (Claude, o-series)
    thinking_budget=None,       # Max thinking tokens
    top_p=None,                 # Nucleus sampling
    top_k=None,                 # Top-k sampling
    frequency_penalty=None,     # Frequency penalty (OpenAI)
    presence_penalty=None,      # Presence penalty (OpenAI)
)
```

### AbstractLLMProvider Methods

Every provider implements these methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `await list_models()` | `list[str]` | Available model identifiers |
| `estimate_token_count(text, model)` | `int` | Token estimate without API call |
| `prepare_tool(tool)` | `Any` | Convert `ToolDefinition` to provider format |
| `await generate_text_response(prompt, config)` | `TokenResponse` or `AsyncIterator[TokenResponse]` | Text generation (streaming when `config.stream=True`) |
| `await generate_tool_parameters(prompt, tools, config)` | `ToolCallResponse` | Function/tool calling |
| `await generate_native_response(prompt, tools, config)` | `NativeResponse` | Text + thinking + tool calls combined |
| `await generate_structured_response(prompt, schema, config)` | `StructuredResponse` | JSON-schema-constrained output |
| `await transcribe(audio_path)` | `str` | Audio-to-text transcription |

Cost estimation (non-abstract, returns `None` if pricing unavailable):

| Method | Returns |
|--------|---------|
| `get_cost_per_1k_input_tokens(model)` | `float \| None` |
| `get_cost_per_1k_output_tokens(model)` | `float \| None` |
| `estimate_cost(text, model, output_tokens)` | `float \| None` |

### Response Types

**TokenResponse** -- single response or streaming chunk:

```python
response.content      # str -- text content
response.stop_reason  # str | None -- "end_turn", "max_tokens", "tool_use"
```

**ToolCallResponse** -- tool invocation results:

```python
response.text_content  # str -- any text alongside tool calls
response.tool_calls    # list[ToolCall] -- each has .tool_name, .tool_id, .parameters
response.stop_reason   # str | None
response.usage         # TokenUsage | None
```

**StructuredResponse** -- JSON-schema-constrained output:

```python
response.structured_output  # dict[str, Any] -- parsed JSON
response.raw_output         # str -- raw model text
response.usage              # TokenUsage | None
```

**NativeResponse** -- complete model output:

```python
response.text_content  # str
response.thinking      # list[ThinkingResponse] -- reasoning blocks
response.tool_calls    # list[ToolCall]
response.usage         # TokenUsage | None
```

**TokenUsage** -- token accounting:

```python
usage.input_tokens                  # int
usage.output_tokens                 # int
usage.cache_creation_input_tokens   # int (Anthropic prompt caching)
usage.cache_read_input_tokens       # int
usage.total_tokens                  # property: input + output
```

### ProviderRegistry

Register providers once and resolve by name or model pattern:

```python
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.providers import OpenAIProvider, AnthropicProvider

registry = ProviderRegistry()
registry.register("openai", OpenAIProvider, api_key="sk-...")
registry.register("anthropic", AnthropicProvider, api_key="sk-ant-...")

# Get by name
provider = registry.get("openai")

# Auto-resolve from model name (gpt-* -> openai, claude-* -> anthropic, etc.)
provider = registry.get_for_model("gpt-4o")
provider = registry.get_for_model("claude-sonnet-4-20250514")

# List registered providers
registry.list_providers()  # ["anthropic", "openai"]
```

Model-to-provider inference patterns: `gpt-*`/`o1-*`/`o3-*` -> openai, `claude-*` -> anthropic, `gemini-*` -> google, `llama-*`/`mixtral-*` -> groq, `grok-*` -> xai.

### Token Counting and Cost Estimation

```python
from quartermaster_providers.providers import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")

tokens = provider.estimate_token_count("Hello, world!", "gpt-4o")
print(f"Estimated tokens: {tokens}")

cost = provider.estimate_cost("Hello, world!", "gpt-4o", output_tokens=100)
if cost is not None:
    print(f"Estimated cost: ${cost:.6f}")
```

## Error Handling

All providers raise consistent exceptions from `quartermaster_providers.exceptions`:

```python
from quartermaster_providers.exceptions import (
    ProviderError,          # Base exception (has .provider, .status_code)
    AuthenticationError,    # Invalid/missing API key (401)
    RateLimitError,         # Rate limited (429, has .retry_after)
    InvalidModelError,      # Model not available (404, has .model)
    InvalidRequestError,    # Malformed request (400)
    ContentFilterError,     # Blocked by safety filter (400)
    ContextLengthError,     # Input exceeds context window (400)
    ServiceUnavailableError,  # Provider temporarily down (503)
)
```

## Testing

Use `MockProvider` for unit tests without real API calls:

```python
from quartermaster_providers import LLMConfig, TokenResponse
from quartermaster_providers.testing import MockProvider

mock = MockProvider(responses=[
    TokenResponse(content="Paris", stop_reason="end_turn"),
    TokenResponse(content="Berlin", stop_reason="end_turn"),
])

config = LLMConfig(model="mock", provider="mock")

response = await mock.generate_text_response("Capital of France?", config)
assert response.content == "Paris"
assert mock.call_count == 1
assert mock.last_prompt == "Capital of France?"

# InMemoryHistory for conversation testing
from quartermaster_providers.testing import InMemoryHistory

history = InMemoryHistory()
history.add_message("user", "Hello")
history.add_message("assistant", "Hi there!")
assert len(history) == 2
```

## Contributing

Contributions welcome. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
