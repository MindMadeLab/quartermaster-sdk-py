# qm-providers

Unified multi-LLM provider abstraction for Python. Write once, run against OpenAI, Anthropic, Google, Groq, xAI, and custom providers.

## What It Does

`qm-providers` provides a single, consistent interface to interact with multiple Large Language Models. Instead of learning each provider's SDK, you define your request once and swap providers without code changes.

```python
from qm_providers import LLMConfig, OpenAIProvider

# Configure once
config = LLMConfig(
    model="gpt-4",
    provider="openai",
    temperature=0.7,
    max_output_tokens=2048,
)

# Use any provider with the same code
provider = OpenAIProvider(api_key="sk-...")
response = provider.generate_text_response(
    prompt="What is AI?",
    config=config,
)
print(response.content)
```

## Installation

Install the core library:

```bash
pip install qm-providers
```

Then add support for the providers you need:

```bash
# Single provider
pip install qm-providers[openai]

# Multiple providers
pip install qm-providers[openai,anthropic,google]

# All providers
pip install qm-providers[all]
```

## Supported Providers

| Provider | Status | Optional Dependency | Notes |
|----------|--------|---------------------|-------|
| OpenAI | ✓ | `openai>=1.0` | gpt-4, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | ✓ | `anthropic>=0.30` | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| Google | ✓ | `google-generativeai>=0.5` | gemini-pro, gemini-1.5-pro |
| Groq | ✓ | `groq>=0.5` | mixtral-8x7b, llama-2-70b |
| xAI | Planned | `xai>=0.1` | grok-1 (when available) |
| Custom | ✓ | - | Implement `AbstractLLMProvider` |

## Features

- **Streaming**: Async generators for streaming responses
- **Tool Calling**: Unified interface for function/tool invocation
- **Structured Output**: JSON schema-based response generation
- **Vision**: Image understanding across supported providers
- **Extended Thinking**: Longer reasoning chains (Claude, OpenAI)
- **Transcription**: Audio-to-text (OpenAI Whisper integration)
- **Token Counting**: Estimate costs before making requests
- **Type Safe**: Full type hints with dataclass responses

## Quick Start

### Text Generation

```python
from qm_providers import LLMConfig, AnthropicProvider

config = LLMConfig(
    model="claude-3-sonnet-20240229",
    provider="anthropic",
    temperature=0.5,
)

provider = AnthropicProvider(api_key="sk-ant-...")
response = provider.generate_text_response(
    prompt="Explain machine learning",
    config=config,
)
print(response.content)
```

### Tool Calling

```python
from qm_providers import ToolDefinition

# Define available tools
tools = [
    ToolDefinition(
        name="get_weather",
        description="Get weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    ),
]

# Generate with tool use
response = provider.generate_tool_parameters(
    prompt="What's the weather in San Francisco?",
    tools=tools,
    config=config,
)

for tool_call in response.tool_calls:
    print(f"Call {tool_call.tool_name} with {tool_call.parameters}")
```

### Streaming Responses

```python
config = LLMConfig(model="gpt-4", provider="openai", stream=True)

async for token_response in provider.generate_text_response(
    prompt="Write a short story",
    config=config,
):
    print(token_response.content, end="", flush=True)
```

### Structured Output

```python
from typing import TypedDict

class Article(TypedDict):
    title: str
    summary: str
    topics: list[str]

response = provider.generate_structured_response(
    prompt="Analyze this article...",
    response_schema=Article,
    config=config,
)

print(response.structured_output.title)
```

## Configuration

`LLMConfig` controls behavior across all providers:

```python
from qm_providers import LLMConfig

config = LLMConfig(
    model="gpt-4",  # Provider's model name
    provider="openai",  # 'openai', 'anthropic', 'google', 'groq', etc.
    stream=False,  # Enable streaming responses
    temperature=0.7,  # 0.0 (deterministic) to 2.0 (creative)
    system_message="You are a helpful assistant",
    max_input_tokens=8000,  # Limit input length
    max_output_tokens=2048,  # Limit output length
    max_messages=10,  # For conversation context
    vision=False,  # Enable image understanding
    thinking_enabled=False,  # Extended thinking (claude, o1)
    thinking_budget=10000,  # Max thinking tokens
)
```

## Creating Custom Providers

Extend `AbstractLLMProvider` to add a new provider:

```python
from qm_providers import AbstractLLMProvider, TokenResponse, LLMConfig

class MyProvider(AbstractLLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def list_models(self) -> list[str]:
        return ["my-model-1", "my-model-2"]

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        # Call your API
        response = await self._call_api(prompt, config)
        return TokenResponse(
            content=response.text,
            stop_reason="end_turn",
        )

    # Implement remaining abstract methods...
    def estimate_token_count(self, text: str, model: str) -> int:
        return len(text.split())

    async def generate_tool_parameters(self, ...): ...
    async def generate_structured_response(self, ...): ...
    async def generate_native_response(self, ...): ...
    async def transcribe(self, audio_path: str) -> str: ...
```

## Architecture

### Core Types

- **LLMConfig** — Request configuration (model, temperature, max_tokens, etc.)
- **TokenResponse** — Single response unit with content and stop_reason
- **ToolCallResponse** — Tool invocation results
- **StructuredResponse** — Validated JSON output
- **NativeResponse** — Hybrid response with text, tool calls, and thinking
- **ToolDefinition** — Tool/function schema
- **MessageHistory** — Conversation protocol for multi-turn

### Provider Pattern

All providers inherit from `AbstractLLMProvider` and implement 8 methods:

1. `list_models()` — Available models
2. `estimate_token_count()` — Pre-request token estimation
3. `prepare_tool()` — Transform tool definitions to provider format
4. `generate_text_response()` — Plain text generation
5. `generate_tool_parameters()` — Function calling
6. `generate_native_response()` — Text + tools + thinking
7. `generate_structured_response()` — JSON schema compliance
8. `transcribe()` — Audio-to-text

### Streaming

Streaming is implicit via async generators:

```python
# Non-streaming returns a single TokenResponse
response = await provider.generate_text_response(prompt, config)

# Streaming returns AsyncIterator[TokenResponse]
config.stream = True
async for chunk in await provider.generate_text_response(prompt, config):
    print(chunk.content)
```

## Token Counting & Cost Estimation

Estimate tokens and cost before making requests:

```python
from qm_providers import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")

# Token count
tokens = provider.estimate_token_count("Hello world", "gpt-4")
print(f"Tokens: {tokens}")

# Cost estimation (requires pricing configuration)
cost_usd = provider.estimate_cost("Hello world", "gpt-4")
print(f"Cost: ${cost_usd:.4f}")
```

## Error Handling

All providers raise consistent exceptions:

```python
from qm_providers.exceptions import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidModelError,
)

try:
    response = provider.generate_text_response(prompt, config)
except AuthenticationError:
    print("Check API keys")
except RateLimitError:
    print("Rate limited, retry with backoff")
except InvalidModelError:
    print("Model not available for this provider")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Testing

Mock providers for unit testing:

```python
from qm_providers.testing import MockProvider

mock = MockProvider(responses=[
    TokenResponse(content="Test response 1"),
    TokenResponse(content="Test response 2"),
])

response = await mock.generate_text_response("test", config)
assert response.content == "Test response 1"
```

## License

Apache License 2.0 — see LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md

## Support

- Documentation: https://quartermaster.ai/docs/providers
- Issues: https://github.com/quartermaster-ai/quartermaster/issues
- Discussions: https://github.com/quartermaster-ai/quartermaster/discussions
