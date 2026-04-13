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

from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama local inference.

    Default endpoint: ``http://localhost:11434/v1``

    Ollama doesn't require an API key.  Just run ``ollama serve`` and
    pull a model (``ollama pull llama3.1``).

    Example::

        provider = OllamaProvider()                       # localhost
        provider = OllamaProvider(base_url="http://gpu-box:11434/v1")
    """

    PROVIDER_NAME = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
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
