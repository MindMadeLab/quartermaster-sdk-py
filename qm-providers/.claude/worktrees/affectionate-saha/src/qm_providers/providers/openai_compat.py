"""Generic OpenAI-compatible provider.

Supports any endpoint that implements the OpenAI Chat Completions API
(Ollama, vLLM, LiteLLM, Together AI, etc.). This is the "bring your own model"
provider.
"""

from __future__ import annotations

from typing import Any

from qm_providers.providers.openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """Generic provider for any OpenAI-compatible API endpoint.

    Use this for Ollama, vLLM, LiteLLM, Together AI, or any other service
    that implements the OpenAI Chat Completions API format.

    Args:
        base_url: The API endpoint base URL (e.g., "http://localhost:11434/v1").
        api_key: API key. Use "ollama" or any string for services that don't require auth.
        auth_method: Authentication method: "bearer" (default), "basic", or "none".
        auth_credentials: For "basic" auth, a tuple of (username, password).
        provider_name: A custom name for this provider instance.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "no-key",
        auth_method: str = "bearer",
        auth_credentials: tuple[str, str] | None = None,
        provider_name: str = "openai-compatible",
    ):
        self.auth_method = auth_method
        self.auth_credentials = auth_credentials
        self.PROVIDER_NAME = provider_name

        if auth_method == "basic" and auth_credentials:
            import base64

            basic_token = base64.b64encode(
                f"{auth_credentials[0]}:{auth_credentials[1]}".encode()
            ).decode()
            super().__init__(api_key=basic_token, base_url=base_url)
        elif auth_method == "none":
            super().__init__(api_key="no-key", base_url=base_url)
        else:
            super().__init__(api_key=api_key, base_url=base_url)

    def _get_client(self):
        if self._client is None:
            import openai

            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
            }
            if self.auth_method == "basic" and self.auth_credentials:
                import httpx

                kwargs["http_client"] = httpx.AsyncClient(
                    auth=(self.auth_credentials[0], self.auth_credentials[1]),
                )
                kwargs["api_key"] = "unused"

            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def list_models(self) -> list[str]:
        """List models from the remote endpoint."""
        try:
            client = self._get_client()
            models = await client.models.list()
            return sorted([m.id for m in models.data])
        except Exception:
            return []

    def estimate_token_count(self, text: str, model: str) -> int:
        """Rough estimation — tiktoken may not work for custom models."""
        return int(len(text.split()) * 1.3)

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        return None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        return None
