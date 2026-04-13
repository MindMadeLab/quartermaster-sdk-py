# Quartermaster API — OpenAI-Compatible Proxy Endpoint

## Overview

Quartermaster open-source needs to connect to the **Quartermaster Cloud** service
so that users can access all LLM providers (OpenAI, Anthropic, Google, Groq, xAI)
through a **single API key** instead of managing separate keys for each provider.

The Quartermaster Cloud exposes an **OpenAI-compatible endpoint** — meaning any
tool, SDK, or library that speaks the OpenAI Chat Completions API can be pointed
at `https://api.quartermaster.ai/v1` and it will "just work".

---

## What We Need in the Quartermaster Cloud Backend

### 1. OpenAI-Compatible Chat Completions Endpoint

```
POST https://api.quartermaster.ai/v1/chat/completions
Authorization: Bearer qm-xxxxxxxxxxxxxxxx
```

**Must support the full OpenAI Chat Completions spec:**

| Feature | Status | Notes |
|---------|--------|-------|
| `model` parameter | Required | Maps to actual provider model (e.g. `gpt-4o` → OpenAI, `claude-sonnet-4-20250514` → Anthropic) |
| `messages` array | Required | Standard `role`/`content` format |
| `temperature`, `top_p`, `max_tokens` | Required | Pass-through to underlying provider |
| `tools` / `tool_choice` | Required | Function calling / tool use |
| `response_format` | Required | JSON mode / structured output |
| `stream` | Required | SSE streaming with `data: {...}` chunks |
| `stop` sequences | Optional | |
| `frequency_penalty`, `presence_penalty` | Optional | Only for providers that support them |
| Thinking/reasoning (`thinking` param) | Extension | For Claude models with extended thinking |

**Request format** (identical to OpenAI):
```json
{
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": false,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
}
```

**Response format** (identical to OpenAI):
```json
{
    "id": "qm-abc123",
    "object": "chat.completion",
    "created": 1713000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 10,
        "total_tokens": 35
    }
}
```

### 2. Models List Endpoint

```
GET https://api.quartermaster.ai/v1/models
Authorization: Bearer qm-xxxxxxxxxxxxxxxx
```

Returns all models available to the user based on their plan:
```json
{
    "object": "list",
    "data": [
        {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
        {"id": "claude-sonnet-4-20250514", "object": "model", "owned_by": "anthropic"},
        {"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"}
    ]
}
```

### 3. Authentication & API Keys

```
POST https://api.quartermaster.ai/v1/auth/keys
Authorization: Bearer qm-xxxxxxxxxxxxxxxx
```

- API keys prefixed with `qm-` to distinguish from provider keys
- Keys scoped to organization/team
- Rate limiting per key
- Usage tracking per key (tokens, requests, cost)

### 4. Usage & Billing Endpoint

```
GET https://api.quartermaster.ai/v1/usage
Authorization: Bearer qm-xxxxxxxxxxxxxxxx
```

Returns token usage, cost breakdown by model/provider, and billing info.

---

## What We Need in Quartermaster Open-Source

### 1. `QuartermasterProvider` in `quartermaster-providers`

A new provider implementation at:
```
quartermaster-providers/src/quartermaster_providers/providers/quartermaster.py
```

This provider:
- Uses the OpenAI SDK pointed at `https://api.quartermaster.ai/v1`
- Authenticates with `QUARTERMASTER_API_KEY` env var or constructor param
- Supports ALL models from ALL providers through the single endpoint
- Inherits from `AbstractLLMProvider`
- Registers as `"quartermaster"` in the provider registry

```python
from quartermaster_providers import ProviderRegistry

registry = ProviderRegistry()
registry.register("quartermaster", QuartermasterProvider, api_key="qm-xxx")

# Now use ANY model through Quartermaster
provider = registry.get_for_model("gpt-4o")       # routed through QM
provider = registry.get_for_model("claude-sonnet-4-20250514")  # also through QM
```

### 2. Auto-Configuration via Environment Variable

When `QUARTERMASTER_API_KEY` is set, the default registry should automatically
register the QuartermasterProvider as the primary provider for all models:

```python
# In registry.py or a new auto_config.py
import os

def auto_configure_registry(registry: ProviderRegistry) -> None:
    api_key = os.environ.get("QUARTERMASTER_API_KEY")
    if api_key:
        from quartermaster_providers.providers.quartermaster import QuartermasterProvider
        registry.register("quartermaster", QuartermasterProvider, api_key=api_key)
```

### 3. CLI Configuration

A simple way to configure the API key:
```bash
export QUARTERMASTER_API_KEY=qm-xxxxxxxxxxxxxxxx
# or
echo "QUARTERMASTER_API_KEY=qm-xxx" >> .env
```

---

## Provider Routing Logic (Backend)

The Quartermaster Cloud backend routes requests based on model name:

| Model Pattern | Provider | Backend Action |
|---------------|----------|----------------|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI | Forward to `api.openai.com/v1` |
| `claude-*` | Anthropic | Translate to Anthropic Messages API |
| `gemini-*` | Google | Translate to Gemini API |
| `llama-*`, `mixtral-*` | Groq | Forward to Groq API |
| `grok-*` | xAI | Forward to xAI API |

**Translation layer responsibilities:**
- Convert OpenAI-format requests → provider-native format
- Convert provider-native responses → OpenAI-format responses
- Handle streaming translation (SSE format differences)
- Map error codes to OpenAI-compatible error responses
- Handle tool/function calling format differences

---

## Implementation Priority

1. **P0**: Chat Completions endpoint (non-streaming) with OpenAI + Anthropic routing
2. **P0**: Models list endpoint
3. **P0**: `QuartermasterProvider` in open-source
4. **P1**: Streaming support
5. **P1**: Tool/function calling support
6. **P1**: Google Gemini routing
7. **P2**: Usage/billing endpoints
8. **P2**: Groq and xAI routing
9. **P2**: Image/vision support
10. **P3**: Audio transcription routing

---

## Security Considerations

- All traffic over HTTPS
- API keys never logged
- Provider API keys stored encrypted in Quartermaster Cloud
- Request/response bodies not stored (only metadata for billing)
- Rate limiting per API key
- IP allowlisting (optional, enterprise feature)
