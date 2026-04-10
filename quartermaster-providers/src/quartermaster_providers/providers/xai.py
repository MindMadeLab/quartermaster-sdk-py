"""xAI LLM provider implementation (Grok).

xAI's Grok models use an OpenAI-compatible API, so this provider
extends OpenAIProvider with xAI-specific defaults and model list.
"""

from __future__ import annotations

from quartermaster_providers.providers.openai import OpenAIProvider

XAI_BASE_URL = "https://api.x.ai/v1"

XAI_MODELS = [
    "grok-2",
    "grok-2-mini",
    "grok-beta",
]

XAI_PRICING: dict[str, dict[str, float]] = {
    "grok-2": {"input": 0.002, "output": 0.01},
    "grok-2-mini": {"input": 0.0002, "output": 0.001},
}


class XAIProvider(OpenAIProvider):
    """xAI provider for Grok models.

    Uses OpenAI-compatible API with xAI's endpoint.

    Args:
        api_key: xAI API key.
    """

    PROVIDER_NAME = "xai"

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url=XAI_BASE_URL)

    async def list_models(self) -> list[str]:
        try:
            return await super().list_models()
        except Exception:
            return list(XAI_MODELS)

    def estimate_token_count(self, text: str, model: str) -> int:
        return int(len(text.split()) * 1.3)

    async def transcribe(self, audio_path: str) -> str:
        from quartermaster_providers.exceptions import ProviderError

        raise ProviderError(
            "xAI does not support audio transcription. Use OpenAIProvider for transcription.",
            provider=self.PROVIDER_NAME,
        )

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        pricing = XAI_PRICING.get(model)
        return pricing["input"] if pricing else None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        pricing = XAI_PRICING.get(model)
        return pricing["output"] if pricing else None
