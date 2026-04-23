"""Generic OpenAI-compatible provider.

Supports any endpoint that implements the OpenAI Chat Completions API
(Ollama, vLLM, LiteLLM, Together AI, etc.). This is the "bring your own model"
provider.
"""

from __future__ import annotations

from typing import Any

from quartermaster_providers.providers.openai import OpenAIProvider


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
        # Per-loop client cache — same rationale as OpenAIProvider._get_client,
        # but this subclass has to build its own httpx.AsyncClient on the same
        # loop as the wrapping openai.AsyncOpenAI (for basic-auth / extra
        # headers), so we can't just delegate to super() — we have to
        # replicate the cache lookup around the extra httpx construction.
        import asyncio

        # Back-compat external injection — see OpenAIProvider._get_client.
        if self._client is not None and not any(
            c is self._client for (_, c) in self._clients_by_loop.values()
        ):
            return self._client

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        dead_keys = [
            key
            for key, (loop, _client) in self._clients_by_loop.items()
            if loop is not None and loop.is_closed()
        ]
        for key in dead_keys:
            self._clients_by_loop.pop(key, None)

        loop_key = id(current_loop) if current_loop is not None else 0
        entry = self._clients_by_loop.get(loop_key)
        if entry is not None and entry[0] is current_loop:
            self._client = entry[1]
            return self._client

        import openai

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }

        extra_headers = getattr(self, "_extra_headers", None)
        needs_http_client = bool(
            (self.auth_method == "basic" and self.auth_credentials) or extra_headers
        )

        if needs_http_client:
            import httpx

            client_kwargs: dict[str, Any] = {}
            if self.auth_method == "basic" and self.auth_credentials:
                client_kwargs["auth"] = (
                    self.auth_credentials[0],
                    self.auth_credentials[1],
                )
                kwargs["api_key"] = "unused"
            if extra_headers:
                client_kwargs["headers"] = dict(extra_headers)
            kwargs["http_client"] = httpx.AsyncClient(**client_kwargs)

        client = openai.AsyncOpenAI(**kwargs)
        self._clients_by_loop[loop_key] = (current_loop, client)
        self._client = client
        return client

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
