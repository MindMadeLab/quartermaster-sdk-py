"""Quartermaster Cloud provider — access all models through a single API key.

Routes all requests through the Quartermaster Cloud proxy, which handles
provider-specific translation (OpenAI, Anthropic, Google, Groq, xAI) behind
a single OpenAI-compatible endpoint.

Usage::

    from quartermaster_providers.providers.quartermaster import QuartermasterProvider

    provider = QuartermasterProvider(api_key="qm-xxxx")
    # or just set QUARTERMASTER_API_KEY env var
    provider = QuartermasterProvider()

    config = LLMConfig(model="gpt-4o", provider="quartermaster")
    response = await provider.generate_text_response("Hello!", config)

    # Works with ANY model — the cloud proxy routes to the correct provider
    config = LLMConfig(model="claude-sonnet-4-20250514", provider="quartermaster")
    response = await provider.generate_text_response("Hello!", config)
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_providers.providers.openai import OpenAIProvider

QUARTERMASTER_API_URL = "https://api.quartermaster.ai/v1"

# All models available through Quartermaster Cloud (union of all providers)
QUARTERMASTER_MODELS = [
    # OpenAI
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
    # Anthropic
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    # Google
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # Groq
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    # xAI
    "grok-2",
    "grok-2-mini",
]


class QuartermasterProvider(OpenAIProvider):
    """Quartermaster Cloud provider — all models, one API key.

    Extends the OpenAI provider pointed at the Quartermaster Cloud proxy.
    All model requests are routed through Quartermaster's API, which
    handles provider-specific translation automatically.

    Args:
        api_key: Quartermaster API key (``qm-...``).
            Falls back to ``QUARTERMASTER_API_KEY`` environment variable.
        base_url: Override the API endpoint (default: ``https://api.quartermaster.ai/v1``).
    """

    PROVIDER_NAME = "quartermaster"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        resolved_key = api_key or os.environ.get("QUARTERMASTER_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Quartermaster API key required. "
                "Pass api_key= or set QUARTERMASTER_API_KEY environment variable. "
                "Get your key at https://app.quartermaster.ai/settings/api-keys"
            )

        resolved_url = base_url or os.environ.get(
            "QUARTERMASTER_API_URL", QUARTERMASTER_API_URL
        )

        super().__init__(
            api_key=resolved_key,
            base_url=resolved_url,
        )

    async def list_models(self) -> list[str]:
        """List all models available through Quartermaster Cloud.

        Attempts to fetch the live model list from the API.
        Falls back to the built-in list if the API is unreachable.
        """
        try:
            client = self._get_client()
            response = await client.models.list()
            return [m.id for m in response.data]
        except Exception:
            return list(QUARTERMASTER_MODELS)

    def estimate_token_count(self, text: str, model: str) -> int:
        """Estimate token count using tiktoken or heuristic fallback."""
        try:
            return super().estimate_token_count(text, model)
        except Exception:
            # Fallback: ~4 chars per token
            return max(1, len(text) // 4)

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        """Cost tracking through Quartermaster — returns None (billed by QM)."""
        return None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        """Cost tracking through Quartermaster — returns None (billed by QM)."""
        return None
