"""Groq LLM provider implementation.

Groq provides fast inference for open-weight models. Its API is
OpenAI-compatible, so this provider extends OpenAIProvider with
Groq-specific defaults and pricing.
"""

from __future__ import annotations

from typing import cast

from quartermaster_providers.providers.openai import OpenAIProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-90b-vision-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

GROQ_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    "gemma2-9b-it": {"input": 0.0002, "output": 0.0002},
}


class GroqProvider(OpenAIProvider):
    """Groq provider for fast inference on open-weight models.

    Uses OpenAI-compatible API with Groq's endpoint.

    Args:
        api_key: Groq API key.
    """

    PROVIDER_NAME = "groq"

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url=GROQ_BASE_URL)

    async def list_models(self) -> list[str]:
        try:
            return await super().list_models()
        except Exception:
            return list(GROQ_MODELS)

    def estimate_token_count(self, text: str, model: str) -> int:
        return int(len(text.split()) * 1.3)

    async def transcribe(self, audio_path: str) -> str:
        """Groq supports Whisper transcription."""
        from pathlib import Path

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        client = self._get_client()
        try:
            with open(audio_path, "rb") as audio_file:
                response = await client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                )
            return cast(str, response.text)
        except Exception as e:
            self._handle_api_error(e)

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        pricing = GROQ_PRICING.get(model)
        return pricing["input"] if pricing else None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        pricing = GROQ_PRICING.get(model)
        return pricing["output"] if pricing else None
