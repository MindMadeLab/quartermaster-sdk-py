# LLM Providers

The `quartermaster-providers` package provides a unified abstraction over multiple LLM providers. All providers implement the same `AbstractLLMProvider` interface, so switching between OpenAI, Anthropic, Google, Groq, or xAI requires only a configuration change.

## Supported Providers

| Provider | Class | Models | Features |
|----------|-------|--------|----------|
| **OpenAI** | `OpenAIProvider` | gpt-4o, gpt-4, o1, o3, chatgpt-4o | Text, streaming, tool calling, structured output, vision, transcription |
| **Anthropic** | `AnthropicProvider` | claude-sonnet-4-20250514, claude-3.5-haiku, claude-3-opus | Text, streaming, tool calling, extended thinking, vision |
| **Google** | `GoogleProvider` | gemini-2.5-pro, gemini-2.0-flash, gemma-* | Text, streaming, tool calling, structured output |
| **Groq** | `GroqProvider` | llama-*, mixtral-* | Text, streaming, tool calling (fast inference) |
| **xAI** | `XAIProvider` | grok-* | Text, streaming, tool calling |
| **OpenAI-Compatible** | `OpenAICompatProvider` | Any OpenAI-compatible API | Ollama, vLLM, LiteLLM, Together AI, etc. |
| **Ollama** | `OllamaProvider` | llama3, mistral, phi, qwen, etc. | Local inference, no API key required (`http://localhost:11434/v1`) |
| **vLLM** | `VLLMProvider` | Any HuggingFace model | High-throughput GPU serving (`http://localhost:8000/v1`) |
| **LM Studio** | `LMStudioProvider` | Any GGUF model | Desktop app with local server (`http://localhost:1234/v1`) |
| **TGI** | `TGIProvider` | Any HuggingFace model | Hugging Face Text Generation Inference (`http://localhost:8080/v1`) |
| **LocalAI** | `LocalAIProvider` | Any supported model | Drop-in OpenAI replacement (`http://localhost:8080/v1`) |
| **llama.cpp** | `LlamaCppProvider` | Any GGUF model | Lightweight C++ inference server (`http://localhost:8080/v1`) |

## Local / Self-Hosted Providers

Quartermaster supports six local inference engines out of the box. All of them expose an OpenAI-compatible `/v1` endpoint and are wrapped as thin subclasses of `OpenAICompatibleProvider`. No external API keys are required -- just point to your running server.

### Supported Engines

| Engine | Registry name | Default URL | Notes |
|--------|--------------|-------------|-------|
| Ollama | `ollama` | `http://localhost:11434/v1` | No API key needed. Run `ollama serve` and `ollama pull <model>`. |
| vLLM | `vllm` | `http://localhost:8000/v1` | GPU-accelerated. Start with `vllm serve <model>`. Optional API key via `--api-key`. |
| LM Studio | `lm-studio` | `http://localhost:1234/v1` | Enable the local server in LM Studio's Developer tab. |
| TGI | `tgi` | `http://localhost:8080/v1` | Hugging Face Text Generation Inference. Run via Docker. |
| LocalAI | `localai` | `http://localhost:8080/v1` | Drop-in OpenAI replacement for CPU and GPU. |
| llama.cpp | `llama-cpp` | `http://localhost:8080/v1` | Start with `llama-server -m model.gguf --port 8080`. |

### One-Liner Registration

Use `register_local()` to register any engine with sensible defaults:

```python
from quartermaster_providers import ProviderRegistry

registry = ProviderRegistry(auto_configure=False)
registry.register_local("ollama")
```

Override the default URL or supply an API key when the server is on another host:

```python
registry.register_local(
    "vllm",
    base_url="http://gpu-box:8000/v1",
    api_key="my-vllm-key",
)
```

### Custom Model Patterns

By default, model names like `llama-3-70b` resolve to the Groq cloud provider. To route open-source model names to a local engine instead, add custom patterns:

```python
registry.add_model_pattern(r"llama3.*", "ollama")
registry.add_model_pattern(r"mistral.*", "ollama")
registry.add_model_pattern(r"codellama.*", "ollama")
```

You can also pass patterns directly during registration:

```python
registry.register_local(
    "vllm",
    base_url="http://gpu-box:8000/v1",
    models=[r"llama3.*", r"mistral.*", r"codellama.*"],
)
```

### Default Provider

Set a catch-all provider for any model name that does not match a pattern:

```python
registry.set_default_provider("vllm")
```

Or pass `default=True` during registration:

```python
registry.register_local("vllm", default=True)
```

### Model Resolution Order

When `get_for_model()` is called, the registry resolves the provider in this order:

1. **Custom patterns** -- added via `add_model_pattern()` or `register_local(..., models=[...])`. Checked first, so they can override built-in defaults.
2. **Built-in patterns** -- the global `MODEL_PATTERNS` table (OpenAI, Anthropic, Google, Groq, xAI).
3. **Default provider** -- set via `set_default_provider()` or `register_local(..., default=True)`.
4. **Default provider** -- set via `set_default_provider()` or `register_local(..., default=True)`.

If none of these resolve, an `InvalidModelError` is raised.

### Setup Examples

**Ollama only** -- simplest local setup, one line:

```python
registry = ProviderRegistry(auto_configure=False)
registry.register_local("ollama")
registry.add_model_pattern(r"llama3.*", "ollama")
registry.add_model_pattern(r"mistral.*", "ollama")

provider = registry.get_for_model("llama3.1:70b")  # -> OllamaProvider
```

**Mixed cloud + local** -- cloud for complex reasoning, local for fast inference:

```python
from quartermaster_providers.providers import OpenAIProvider

registry = ProviderRegistry(auto_configure=False)
registry.register("openai", OpenAIProvider, api_key="sk-...")
registry.register_local(
    "vllm",
    base_url="http://gpu-box:8000/v1",
    models=[r"llama3.*", r"mistral.*"],
)

provider = registry.get_for_model("gpt-4o")        # -> OpenAIProvider
provider = registry.get_for_model("llama3.1:70b")   # -> VLLMProvider
```

**Fully private** -- all traffic stays on your infrastructure:

```python
registry = ProviderRegistry(auto_configure=False)

# Main inference via vLLM cluster (catch-all)
registry.register_local(
    "vllm",
    base_url="http://vllm-cluster.internal:8000/v1",
    api_key="internal-key",
    default=True,
)

# Embeddings via Ollama
registry.register_local(
    "ollama",
    base_url="http://ollama.internal:11434/v1",
    models=[r"nomic-embed.*", r"all-minilm.*"],
)

# Custom corporate gateway
registry.register_local(
    "custom",
    base_url="https://llm-gateway.corp.internal/v1",
    name="corp-gateway",
    api_key="corp-secret",
    models=[r"corp-.*"],
)
```

## ProviderRegistry

The `ProviderRegistry` manages provider instances with lazy initialization and automatic model-to-provider inference.

### Registration

```python
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.providers.openai import OpenAIProvider
from quartermaster_providers.providers.anthropic import AnthropicProvider

registry = ProviderRegistry()

# Register with constructor arguments (lazy -- instantiated on first use)
registry.register("openai", OpenAIProvider, api_key="sk-...")
registry.register("anthropic", AnthropicProvider, api_key="sk-ant-...")
```

### Pre-Created Instances

```python
provider = OpenAIProvider(api_key="sk-...", organization="org-...")
registry.register_instance("openai", provider)
```

### Lookup

```python
# Explicit lookup by name
provider = registry.get("openai")

# Automatic inference from model name
provider = registry.get_for_model("gpt-4o")       # resolves to "openai"
provider = registry.get_for_model("claude-sonnet-4-20250514")  # resolves to "anthropic"
provider = registry.get_for_model("gemini-2.5-pro") # resolves to "google"
provider = registry.get_for_model("llama-3-70b")   # resolves to "groq"
provider = registry.get_for_model("grok-2")        # resolves to "xai"
```

### Model Inference Patterns

The registry uses regex patterns to map model names to providers:

| Pattern | Provider |
|---------|----------|
| `gpt-*`, `o1-*`, `o3-*`, `chatgpt-*`, `dall-e*`, `whisper*`, `tts-*` | openai |
| `claude-*` | anthropic |
| `gemini-*`, `gemma-*` | google |
| `llama-*`, `mixtral-*` | groq |
| `grok-*` | xai |

### Global Default Registry

A module-level default registry is available for convenience:

```python
from quartermaster_providers.registry import get_default_registry

registry = get_default_registry()
registry.register("openai", OpenAIProvider, api_key="sk-...")
```

## LLMConfig

The `LLMConfig` dataclass unifies configuration across all providers:

```python
from quartermaster_providers.config import LLMConfig

config = LLMConfig(
    model="gpt-4o",
    provider="openai",
    stream=True,
    temperature=0.7,
    system_message="You are a helpful assistant.",
    max_input_tokens=4096,
    max_output_tokens=1024,
    max_messages=50,
)

# Validate before use
config.validate()
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier (e.g., `"gpt-4o"`, `"claude-sonnet-4-20250514"`) |
| `provider` | `str` | required | Provider name (e.g., `"openai"`, `"anthropic"`) |
| `stream` | `bool` | `False` | Enable token-by-token streaming |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0) |
| `system_message` | `str\|None` | `None` | System prompt to set model behavior |
| `max_input_tokens` | `int\|None` | `None` | Maximum input tokens |
| `max_output_tokens` | `int\|None` | `None` | Maximum output tokens |
| `max_messages` | `int\|None` | `None` | Maximum conversation messages |
| `vision` | `bool` | `False` | Enable vision/image understanding |
| `thinking_enabled` | `bool` | `False` | Enable extended thinking (Claude) |
| `thinking_budget` | `int\|None` | `None` | Max tokens for thinking/reasoning |
| `top_p` | `float\|None` | `None` | Nucleus sampling parameter |
| `top_k` | `int\|None` | `None` | Top-k sampling parameter |
| `frequency_penalty` | `float\|None` | `None` | Penalty for repeating tokens (OpenAI) |
| `presence_penalty` | `float\|None` | `None` | Penalty for new topics (OpenAI) |

### Creating from Dictionaries

```python
config = LLMConfig.from_dict({
    "model": "gpt-4o",
    "provider": "openai",
    "temperature": 0.3,
    "max_output_tokens": 2048,
})
```

## AbstractLLMProvider Interface

All providers implement these methods:

```python
class AbstractLLMProvider(ABC):
    # Discovery
    async def list_models(self) -> list[str]: ...

    # Text generation
    async def generate_text_response(
        self, prompt: str, config: LLMConfig
    ) -> TokenResponse | AsyncIterator[TokenResponse]: ...

    # Tool/function calling
    async def generate_tool_parameters(
        self, prompt: str, tools: list[ToolDefinition], config: LLMConfig
    ) -> ToolCallResponse: ...

    # Complete response (text + thinking + tool calls)
    async def generate_native_response(
        self, prompt: str, tools: list[ToolDefinition] | None, config: LLMConfig | None
    ) -> NativeResponse: ...

    # Structured JSON output
    async def generate_structured_response(
        self, prompt: str, response_schema: dict | type, config: LLMConfig
    ) -> StructuredResponse: ...

    # Audio transcription
    async def transcribe(self, audio_path: str) -> str: ...

    # Token estimation
    def estimate_token_count(self, text: str, model: str) -> int: ...

    # Tool format conversion
    def prepare_tool(self, tool: ToolDefinition) -> Any: ...
```

## Streaming vs Non-Streaming

When `config.stream=True`, `generate_text_response()` returns an `AsyncIterator[TokenResponse]` that yields tokens as they arrive:

```python
config = LLMConfig(model="gpt-4o", provider="openai", stream=True)
provider = registry.get("openai")

response = await provider.generate_text_response("Tell me a story", config)

# Streaming: iterate over tokens
async for token in response:
    print(token.content, end="", flush=True)
    if token.stop_reason:
        print(f"\n[stopped: {token.stop_reason}]")
```

When `config.stream=False` (default), it returns a single `TokenResponse`:

```python
config = LLMConfig(model="gpt-4o", provider="openai", stream=False)
response = await provider.generate_text_response("Tell me a story", config)
print(response.content)
```

## Response Types

| Type | Fields | Use Case |
|------|--------|----------|
| `TokenResponse` | `content`, `stop_reason` | Simple text generation |
| `ToolCallResponse` | `text_content`, `tool_calls`, `stop_reason`, `usage` | Function/tool calling |
| `NativeResponse` | `text_content`, `thinking`, `tool_calls`, `stop_reason`, `usage` | Full response with thinking |
| `StructuredResponse` | `structured_output`, `raw_output`, `stop_reason`, `usage` | JSON schema-guided output |
| `TokenUsage` | `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` | Cost tracking |

## Cost Estimation

Providers expose cost estimation methods:

```python
provider = registry.get("openai")

# Estimate token count
tokens = provider.estimate_token_count("Hello, how are you?", "gpt-4o")

# Estimate cost (input only)
cost = provider.estimate_cost("Hello, how are you?", "gpt-4o")

# Estimate cost (input + expected output)
cost = provider.estimate_cost("Hello, how are you?", "gpt-4o", output_tokens=500)

# Get per-1K-token pricing
input_price = provider.get_cost_per_1k_input_tokens("gpt-4o")
output_price = provider.get_cost_per_1k_output_tokens("gpt-4o")
```

## Adding a Custom Provider

Implement `AbstractLLMProvider` and register it:

```python
from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.config import LLMConfig
from quartermaster_providers.types import TokenResponse, NativeResponse

class MyCustomProvider(AbstractLLMProvider):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    async def list_models(self) -> list[str]:
        return ["my-model-v1", "my-model-v2"]

    def estimate_token_count(self, text: str, model: str) -> int:
        return len(text.split()) * 2  # rough estimate

    def prepare_tool(self, tool):
        return tool  # pass through if using OpenAI-compatible format

    async def generate_text_response(self, prompt, config):
        # Your API call here
        ...

    async def generate_tool_parameters(self, prompt, tools, config):
        ...

    async def generate_native_response(self, prompt, tools=None, config=None):
        ...

    async def generate_structured_response(self, prompt, response_schema, config):
        ...

    async def transcribe(self, audio_path):
        raise NotImplementedError("Transcription not supported")

# Register
registry.register("my-provider", MyCustomProvider, api_key="...", base_url="https://...")
```

For OpenAI-compatible APIs (Ollama, vLLM, LiteLLM), use the built-in `OpenAICompatProvider` instead of writing from scratch.

## See Also

- [Architecture](architecture.md) -- How providers fit into the system
- [Tools](tools.md) -- Tool definitions that providers use for function calling
- [Engine](engine.md) -- How the engine calls providers during execution
