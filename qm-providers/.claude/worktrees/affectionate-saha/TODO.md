# qm-providers — Extraction TODO

Unified multi-LLM provider abstraction. One interface to call OpenAI, Anthropic, Google, Groq, xAI, and any OpenAI-compatible provider (Ollama, vLLM, etc.). Supports streaming, tool calling, structured output, vision, extended thinking, and audio transcription.

## Source Files

Extract from `quartermaster/be/providers/llm_service_providers/`:

| Source File | Purpose |
|---|---|
| `abstract.py` | Core abstraction: `AbstractLLMProvider`, `LLMConfig`, response types |
| `OpenAi.py` | OpenAI implementation (GPT-4o, GPT-5, o-series) |
| `Anthropic.py` | Anthropic implementation (Claude Sonnet, Opus, Haiku) |
| `Google.py` | Google implementation (Gemini models) |
| `Groq.py` | Groq implementation (fast inference) |
| `XAi.py` | xAI implementation (Grok models) |
| `ZAi.py` | Custom provider (OpenAI-compatible) |
| `MindMade.py` | Self-hosted provider (OpenAI-compatible + Basic Auth) |
| `instrumented_abstract.py` | Instrumentation/observability wrapper |
| `__init__.py` | Provider registry and selection logic |

Supporting files:
| Source File | Purpose |
|---|---|
| `quartermaster/be/programs/services/__init__.py` | `ProgramContainer` model (used for tool formatting) |
| `quartermaster/be/thoughts/models.py` | `ThoughtMemory` model (message history) |

## Extractability: 8/10

Clean abstraction with well-defined interfaces. Main coupling points are Django's `get_object_or_404`, `ThoughtMemory` for message history, and `ProgramContainer` for tool definitions. All replaceable with dataclasses/protocols.

## Phase 1: Core Abstraction

### 1.1 Extract Base Types
- [x] Extract `LLMConfig` class — remove Django dependencies, make pure dataclass
  - Fields: model, provider, stream, temperature, system_message, max_input_tokens, max_output_tokens, max_messages, vision, thinking_enabled, thinking_budget
- [x] Extract response types as standalone dataclasses:
  - `TokenResponse(content: str, finish_reason: Optional[str])`
  - `ThinkingResponse(content: str)`
  - `TokenUsage(input_tokens: int, output_tokens: int, cache_read_input_tokens: int, cache_creation_input_tokens: int, thinking_tokens: int)`
  - `ToolCallResponse(tool_name: str, tool_call_id: str, tool_input: dict)`
  - `StructuredResponse` (for JSON-mode outputs)
  - `NativeResponse` (union type for mixed streaming)
- [x] Extract `IMAGE_TOKEN_ESTIMATE` constant

### 1.2 Extract Abstract Provider
- [x] Copy `AbstractLLMProvider` ABC
- [x] Define 8 abstract methods:
  1. `list_models() -> list[str]`
  2. `estimate_token_count(messages, tools) -> int`
  3. `prepare_tool(program) -> dict` — replace `ProgramContainer` with own `ToolDefinition` dataclass
  4. `generate_text_response(config, messages, tools) -> Generator[TokenResponse | ThinkingResponse | TokenUsage]`
  5. `generate_tool_parameters(config, messages, tools) -> Generator[...]`
  6. `generate_native_response(config, messages, tools) -> Generator[...]`
  7. `generate_structured_response(config, messages, response_schema) -> StructuredResponse`
  8. `transcribe(audio_file) -> str`
- [x] Replace `ThoughtMemory` dependency with a `Message` protocol/dataclass
- [x] Replace `ProgramContainer` with `ToolDefinition`

### 1.3 Remove Django Dependencies
- [x] Remove `from django.shortcuts import get_object_or_404`
- [x] Remove `from thoughts.models import ThoughtMemory`
- [x] Remove `from programs.services import ProgramContainer`
- [x] Remove any `ServiceModel` / pricing lookups
- [x] Remove user authentication context
- [x] Remove billing/credit checks (those belong in platform)

## Phase 2: Provider Implementations

### 2.1 OpenAI Provider
- [x] Extract from `OpenAi.py`
- [x] Dependencies: `openai` SDK
- [x] Features to port:
  - Streaming text generation
  - Tool calling (function calling)
  - Structured output (JSON mode / response_format)
  - Vision (image URLs + base64)
  - Audio transcription (Whisper)
  - Token estimation via tiktoken
  - o-series reasoning models support
- [x] Remove QM-specific: billing hooks, user subscription checks
- [x] Make API key configurable via constructor or env var

### 2.2 Anthropic Provider
- [x] Extract from `Anthropic.py`
- [x] Dependencies: `anthropic` SDK
- [x] Features to port:
  - Streaming text generation
  - Tool calling
  - Extended thinking (thinking_enabled, thinking_budget)
  - Vision support
  - Cache control (prompt caching)
  - Token estimation
- [x] Remove QM-specific references
- [x] Handle Anthropic-specific message formatting (system as top-level param)

### 2.3 Google Provider
- [x] Extract from `Google.py`
- [x] Dependencies: `google-generativeai` SDK
- [x] Features to port:
  - Streaming text generation
  - Tool calling
  - Structured output
  - Vision support
  - Gemini-specific features (grounding, etc.)
- [x] Remove QM-specific references

### 2.4 Groq Provider
- [x] Extract from `Groq.py`
- [x] Dependencies: `groq` SDK (or OpenAI-compatible)
- [x] Features to port:
  - Fast inference streaming
  - Tool calling
- [x] Usually extends OpenAI provider with different base URL

### 2.5 xAI Provider
- [x] Extract from `XAi.py`
- [x] OpenAI-compatible API
- [x] Grok model support

### 2.6 Generic OpenAI-Compatible Provider
- [x] Extract pattern from `MindMade.py` / `ZAi.py`
- [x] Support any OpenAI-compatible endpoint (Ollama, vLLM, LiteLLM, Together, etc.)
- [x] Configurable: base_url, api_key, auth method (Bearer, Basic, none)
- [x] This is the "bring your own model" provider

## Phase 3: Provider Registry

### 3.1 Registration System
- [x] `ProviderRegistry` class — register providers by name
- [x] Auto-discovery of installed providers
- [x] `get_provider(name: str, **config) -> AbstractLLMProvider`
- [x] Default providers: openai, anthropic, google, groq, xai
- [x] Custom provider registration: `registry.register("my-ollama", OllamaProvider)`

### 3.2 Model Resolution
- [x] Map model names to providers (e.g., "gpt-4o" → OpenAI, "claude-sonnet-4-20250514" → Anthropic)
- [x] `infer_provider(model_name: str) -> str` — auto-detect provider from model name
- [x] Allow override: user can force a provider for any model

## Phase 4: Tool Formatting

### 4.1 Universal Tool Schema
- [x] `ToolDefinition` dataclass (name, description, parameters as JSON Schema)
- [x] Each provider's `prepare_tool()` converts to provider-specific format
- [x] OpenAI format: `{"type": "function", "function": {...}}`
- [x] Anthropic format: `{"name": ..., "input_schema": {...}}`
- [x] Google format: `{"function_declarations": [...]}`

## Phase 5: Testing

### 5.1 Unit Tests
- [x] Test `LLMConfig` initialization and defaults
- [x] Test all response type dataclasses
- [x] Test `ToolDefinition` to provider-specific format conversion
- [x] Test provider registry (register, get, auto-detect)
- [x] Test model name → provider inference

### 5.2 Mock Provider Tests
- [x] Create `MockProvider(AbstractLLMProvider)` for testing
- [x] Test streaming generator interface
- [x] Test tool calling flow
- [x] Test structured output flow
- [x] Test error handling (API errors, rate limits, timeouts)

### 5.3 Integration Tests (Optional, need API keys)
- [ ] Test OpenAI real API call
- [ ] Test Anthropic real API call
- [ ] Test Google real API call
- [ ] Test Ollama local model call
- [ ] Mark as `@pytest.mark.integration`, skip in CI without keys

## Phase 6: Documentation

### 6.1 README
- [x] Quick start: pick a provider, send a message, get streaming response
- [x] Provider comparison table (features per provider)
- [x] Configuration guide (API keys, custom endpoints)
- [x] Tool calling example
- [x] Structured output example
- [x] Vision example
- [x] "Bring your own model" guide (Ollama, vLLM)

### 6.2 API Reference
- [x] All public classes and methods documented
- [x] Response type reference
- [x] Provider-specific notes and limitations

## Phase 7: CI/CD & PyPI

- [x] GitHub Actions: lint, typecheck, unit tests
- [x] PyPI package: `quartermaster-providers` or `qm-providers`
- [x] Optional dependencies: `pip install qm-providers[openai]`, `qm-providers[anthropic]`, `qm-providers[google]`, `qm-providers[all]`

## Architecture Notes

### Why This Is Valuable
- LiteLLM is the main competitor but it's bloated (100+ providers, complex)
- This is focused: 5-6 providers, clean abstraction, streaming-first
- Generator-based streaming interface is unique and composable
- Tool calling works identically across providers
- Extended thinking support (Anthropic) is rare in abstractions

### Key Design Decisions
1. Generator-based streaming (not callbacks) — composable with Python itertools
2. Separate `generate_text_response` vs `generate_native_response` — text-only vs mixed mode
3. Provider as class, not function — holds state (API client, config)
4. `ToolDefinition` is provider-agnostic — each provider converts internally
5. Optional dependencies — don't force installing all SDKs
