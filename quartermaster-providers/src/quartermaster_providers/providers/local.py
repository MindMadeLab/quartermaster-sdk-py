"""Local / self-hosted LLM provider implementations.

Convenience wrappers around ``OpenAICompatibleProvider`` for popular
self-hosted inference engines.  Each class ships with sensible defaults
(base URL, auth, provider name) so the user only needs::

    from quartermaster_providers.providers.local import OllamaProvider
    registry.register("ollama", OllamaProvider)
    # or just:
    registry.register_local("ollama")

Supported engines
~~~~~~~~~~~~~~~~~
* **Ollama** — ``http://localhost:11434/v1``
* **vLLM** — ``http://localhost:8000/v1``
* **LM Studio** — ``http://localhost:1234/v1``
* **TGI** (Text Generation Inference) — ``http://localhost:8080/v1``
* **LocalAI** — ``http://localhost:8080/v1``
* **llama.cpp server** — ``http://localhost:8080/v1``

All of these expose an OpenAI-compatible ``/v1`` endpoint.
"""

from __future__ import annotations

import os

from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider


def _normalize_openai_compat_url(base_url: str) -> str:
    """Ensure an OpenAI-compatible endpoint URL ends with ``/v1``.

    Users naturally type ``http://host:11434`` (the bare Ollama address) but
    the OpenAI SDK needs ``/v1`` appended.  We add it iff it isn't already
    there so both forms work.
    """
    if not base_url:
        return base_url
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1") or "/v1/" in stripped:
        return stripped
    return f"{stripped}/v1"


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama local inference.

    Default endpoint: ``http://localhost:11434/v1`` (overridable via the
    ``OLLAMA_HOST`` env var — accepts either ``http://host:port`` or
    ``http://host:port/v1``).

    Ollama doesn't require an API key.  Just run ``ollama serve`` and
    pull a model (``ollama pull llama3.1``).

    Example::

        provider = OllamaProvider()                       # localhost or $OLLAMA_HOST
        provider = OllamaProvider(base_url="http://gpu-box:11434")     # /v1 added
        provider = OllamaProvider(base_url="http://gpu-box:11434/v1")  # already correct
    """

    PROVIDER_NAME = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str = "ollama",
        **kwargs,
    ):
        resolved = base_url or os.environ.get("OLLAMA_HOST") or self.DEFAULT_BASE_URL
        super().__init__(
            base_url=_normalize_openai_compat_url(resolved),
            api_key=api_key,
            auth_method="none",
            provider_name="ollama",
            **kwargs,
        )


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM inference server.

    Default endpoint: ``http://localhost:8000/v1``

    Start vLLM with::

        vllm serve meta-llama/Llama-3.1-8B-Instruct

    Example::

        provider = VLLMProvider()
        provider = VLLMProvider(base_url="http://gpu-cluster:8000/v1",
                                api_key="my-vllm-key")
    """

    PROVIDER_NAME = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "no-key",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            auth_method="bearer" if api_key != "no-key" else "none",
            provider_name="vllm",
            **kwargs,
        )


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio local server.

    Default endpoint: ``http://localhost:1234/v1``

    Enable the local server in LM Studio's Developer tab.

    Example::

        provider = LMStudioProvider()
    """

    PROVIDER_NAME = "lm-studio"

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            auth_method="none",
            provider_name="lm-studio",
            **kwargs,
        )


class TGIProvider(OpenAICompatibleProvider):
    """Hugging Face Text Generation Inference (TGI).

    Default endpoint: ``http://localhost:8080/v1``

    Start TGI with::

        docker run --gpus all -p 8080:80 \\
          ghcr.io/huggingface/text-generation-inference \\
          --model-id meta-llama/Llama-3.1-8B-Instruct

    Example::

        provider = TGIProvider()
        provider = TGIProvider(base_url="http://tgi-server:8080/v1",
                               api_key="hf_...")
    """

    PROVIDER_NAME = "tgi"

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "no-key",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            auth_method="bearer" if api_key != "no-key" else "none",
            provider_name="tgi",
            **kwargs,
        )


class LocalAIProvider(OpenAICompatibleProvider):
    """LocalAI drop-in replacement.

    Default endpoint: ``http://localhost:8080/v1``

    Example::

        provider = LocalAIProvider()
    """

    PROVIDER_NAME = "localai"

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "no-key",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            auth_method="none",
            provider_name="localai",
            **kwargs,
        )


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp HTTP server (``llama-server``).

    Default endpoint: ``http://localhost:8080/v1``

    Start with::

        llama-server -m model.gguf --port 8080

    Example::

        provider = LlamaCppProvider()
    """

    PROVIDER_NAME = "llama-cpp"

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "no-key",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            auth_method="none",
            provider_name="llama-cpp",
            **kwargs,
        )


# ── Lookup table for register_local() ────────────────────────────────

LOCAL_PROVIDERS: dict[str, type[OpenAICompatibleProvider]] = {
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
    "lm-studio": LMStudioProvider,
    "tgi": TGIProvider,
    "localai": LocalAIProvider,
    "llama-cpp": LlamaCppProvider,
}
"""Maps shorthand names to provider classes for :meth:`ProviderRegistry.register_local`."""
