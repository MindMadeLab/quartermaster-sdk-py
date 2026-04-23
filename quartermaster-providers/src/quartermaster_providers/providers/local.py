"""Local / self-hosted LLM provider implementations.

Thin wrappers around :class:`OpenAICompatibleProvider` for popular
self-hosted inference engines. Every engine listed here exposes an
OpenAI-compatible ``/v1`` endpoint, so the same async openai SDK path
powers them all — no per-engine HTTP code, no native transport detours.

Supported engines:

* **Ollama** — ``http://localhost:11434/v1``
* **vLLM** — ``http://localhost:8000/v1``
* **LM Studio** — ``http://localhost:1234/v1``
* **TGI** — ``http://localhost:8080/v1``
* **LocalAI** — ``http://localhost:8080/v1``
* **llama.cpp server** — ``http://localhost:8080/v1``

Usage::

    from quartermaster_providers import register_local
    registry = register_local("ollama", default_model="gemma4:26b")
"""

from __future__ import annotations

import logging
import os
from typing import Any

from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


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


# Hostnames / IPs that point at cloud-metadata or link-local services —
# legitimate Ollama deployments never live here.  We warn rather than
# block so niche tunneling setups keep working, but the warning makes
# misconfigurations visible in logs.
_SSRF_SUSPICIOUS_HOSTS: frozenset[str] = frozenset(
    {
        "169.254.169.254",  # AWS / GCP / Azure / Oracle Cloud IMDS
        "metadata.google.internal",
        "metadata",
        "fd00:ec2::254",  # AWS IPv6 IMDS
        "100.100.100.200",  # Alibaba Cloud metadata
    }
)


def _warn_if_suspicious_url(base_url: str) -> None:
    """Log a warning when *base_url* points at a known SSRF-magnet host."""
    if not base_url:
        return
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").lower()
    if host in _SSRF_SUSPICIOUS_HOSTS:
        logger.warning(
            "OllamaProvider base_url targets %r — this is a known cloud-"
            "metadata / link-local address, not an Ollama instance. If you "
            "intentionally tunnel Ollama through that host you can ignore "
            "this; otherwise check your configuration.",
            host,
        )


def _env_auth() -> tuple[str, str] | None:
    """Read HTTP Basic Auth credentials from OLLAMA_USER + OLLAMA_PASS env vars."""
    user = os.environ.get("OLLAMA_USER")
    pwd = os.environ.get("OLLAMA_PASS")
    if user and pwd:
        return (user, pwd)
    return None


def _env_headers() -> dict[str, str]:
    """Read extra HTTP headers from OLLAMA_HEADERS env var.

    Format: ``Key:Value,Key2:Value2`` (comma-separated ``Key:Value`` pairs).
    """
    raw = os.environ.get("OLLAMA_HEADERS", "")
    if not raw.strip():
        return {}
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, _, v = pair.partition(":")
            headers[k.strip()] = v.strip()
    return headers


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama local inference via the OpenAI-compatible ``/v1`` endpoint.

    Default endpoint: ``http://localhost:11434/v1`` (overridable via the
    ``OLLAMA_HOST`` env var — accepts either ``http://host:port`` or
    ``http://host:port/v1``).

    Ollama doesn't require an API key. Just run ``ollama serve`` and pull
    a model (``ollama pull llama3.1``).

    Example::

        provider = OllamaProvider()                                    # localhost or $OLLAMA_HOST
        provider = OllamaProvider(base_url="http://gpu-box:11434")     # /v1 auto-appended
        provider = OllamaProvider(base_url="http://gpu-box:11434/v1")  # already correct
    """

    PROVIDER_NAME = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str = "ollama",
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        resolved = base_url or os.environ.get("OLLAMA_HOST") or self.DEFAULT_BASE_URL
        _warn_if_suspicious_url(resolved)

        # HTTP Basic Auth + custom headers for Ollama behind reverse
        # proxies (nginx, Caddy, Traefik). Resolution:
        #   auth kwarg > OLLAMA_USER + OLLAMA_PASS env vars
        #   headers kwarg > OLLAMA_HEADERS env var (comma-separated ``Key:Value``)
        http_auth = auth or _env_auth()
        extra_headers = dict(headers or _env_headers())

        auth_method = "bearer"
        auth_credentials: tuple[str, str] | None = None
        if http_auth:
            auth_method = "basic"
            auth_credentials = http_auth
        elif api_key == "ollama":
            # Plain Ollama doesn't check the key; any non-empty string
            # keeps the openai SDK happy.
            auth_method = "none"

        self._extra_headers: dict[str, str] = extra_headers

        super().__init__(
            base_url=_normalize_openai_compat_url(resolved),
            api_key=api_key,
            auth_method=auth_method,
            auth_credentials=auth_credentials,
            provider_name="ollama",
            **kwargs,
        )


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM inference server.

    Default endpoint: ``http://localhost:8000/v1``

    Start vLLM with::

        vllm serve meta-llama/Llama-3.1-8B-Instruct
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


LOCAL_PROVIDERS: dict[str, type[OpenAICompatibleProvider]] = {
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
    "lm-studio": LMStudioProvider,
    "tgi": TGIProvider,
    "localai": LocalAIProvider,
    "llama-cpp": LlamaCppProvider,
}
"""Maps shorthand names to provider classes for :meth:`ProviderRegistry.register_local`."""
