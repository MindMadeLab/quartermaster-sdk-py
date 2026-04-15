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

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from quartermaster_providers.exceptions import ProviderError, ServiceUnavailableError
from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider
from quartermaster_providers.types import ToolCall

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


def _strip_v1(base_url: str) -> str:
    """Drop a trailing ``/v1`` segment.

    The OpenAI-compat path lives at ``/v1/chat/completions`` but the
    native Ollama API lives at ``/api/chat``.  ``OllamaProvider`` stores
    its base URL in the ``/v1`` form (so the openai SDK works), so the
    sync ``chat()`` shim has to peel ``/v1`` back off before talking to
    ``/api/chat``.
    """
    if not base_url:
        return base_url
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        return stripped[: -len("/v1")]
    return stripped


@dataclass
class ChatResult:
    """Result of a synchronous :meth:`OllamaProvider.chat` call.

    Mirrors what downstream integrators were already building by hand on
    top of raw ``requests.post`` against ``/v1/chat/completions``: a
    populated ``content`` string (auto-promoted from a ``thinking``/
    ``reasoning`` field when the model leaves ``content`` empty),
    structured ``tool_calls``, and a usage breakdown.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    raw: dict[str, Any] | None = None


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

    # ── Synchronous native /api/chat shim ────────────────────────────
    #
    # The async ``generate_*`` methods route through the OpenAI SDK against
    # Ollama's ``/v1`` proxy, which is great for graph runs but painful
    # for non-async callers (Celery, Django request-handlers, CLI scripts)
    # — they need ``asgiref.sync.async_to_sync`` wrappers and they lose
    # the reasoning-content surfacing from gemma4-style models.  ``chat()``
    # is the small, sync, native replacement they actually want: hits
    # ``POST {base}/api/chat`` directly via httpx, surfaces ``thinking`` /
    # ``reasoning`` text when ``message.content`` comes back empty, and
    # raises connection errors instead of swallowing them into a
    # ``success=True`` ``FlowResult``.

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
        timeout: float = 120.0,
    ) -> ChatResult:
        """Synchronous one-shot chat against Ollama's native ``/api/chat``.

        Args:
            messages: OpenAI-style ``[{"role": "user", "content": "..."}]``
                list.  ``system``/``user``/``assistant``/``tool`` roles
                pass through verbatim.
            model: Model id (e.g. ``"gemma4:26b"``).  Defaults to the
                provider's bound default model.  Required when no default
                is configured.
            tools: Optional tool definitions in OpenAI function-calling
                shape; passed through to Ollama unchanged.
            temperature: Sampling temperature (Ollama maps it to
                ``options.temperature``).
            max_output_tokens: Hard cap on generated tokens (mapped to
                Ollama's ``options.num_predict``).  Critical for
                reasoning models like ``gemma4:26b`` that otherwise burn
                the default 2 048-token budget on internal thinking
                before producing visible output.
            thinking_level: One of ``off``/``low``/``medium``/``high``,
                or ``None`` to leave it to the model.  Forwarded as
                Ollama's ``think`` parameter.
            timeout: HTTP timeout in seconds.  Connection / read errors
                bubble up as :class:`ServiceUnavailableError`.

        Returns:
            A :class:`ChatResult` with ``content`` always populated when
            the server returned anything visible — ``thinking`` /
            ``reasoning`` fields are promoted to ``content`` when the
            primary ``message.content`` field comes back empty.
        """
        import httpx

        resolved_model = model or self._default_model_for_chat()
        if not resolved_model:
            raise ProviderError(
                "OllamaProvider.chat() requires a model. Pass model=... "
                "or register the provider with a default_model.",
                provider=self.PROVIDER_NAME,
            )

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_output_tokens is not None:
            options["num_predict"] = int(max_output_tokens)

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": list(messages),
            "stream": False,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = list(tools)
        if thinking_level is not None:
            # Ollama's native parameter name is ``think`` (bool/string);
            # we accept the higher-level enum so callers don't have to
            # know which version of Ollama they're talking to.
            payload["think"] = thinking_level not in ("off", False, None)

        url = f"{_strip_v1(self.base_url)}/api/chat"

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            # Status-code errors carry useful body context; surface it.
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                provider=self.PROVIDER_NAME,
                status_code=exc.response.status_code,
            ) from exc
        except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout) as exc:
            # Network unreachable / timed out — caller needs to know
            # this didn't even hit the model, not a soft "no answer".
            raise ServiceUnavailableError(
                f"Could not reach Ollama at {url}: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc
        except Exception as exc:  # pragma: no cover — defensive
            raise ProviderError(
                f"Unexpected error talking to Ollama: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        return _parse_native_chat(body)

    def _default_model_for_chat(self) -> str | None:
        """Best-effort default model lookup for the sync chat shim.

        Ollama doesn't carry a default model on the connection itself —
        the registry holds the engine-level default set via
        ``register_local("ollama", default_model=...)``.  We don't have a
        back-pointer to the registry from inside the provider, so this
        hook just returns ``None`` and lets the caller pass ``model=``.
        Subclasses (or registry-aware factories) may override it.
        """
        return getattr(self, "_chat_default_model", None)


def _parse_native_chat(body: dict[str, Any]) -> ChatResult:
    """Translate Ollama's ``/api/chat`` JSON into a :class:`ChatResult`.

    Handles three classes of model behaviour that downstream integrators
    keep tripping over:

    * Plain models — ``message.content`` is the answer.
    * Reasoning models (``gemma4:26b`` and friends) — ``message.content``
      may be empty while ``message.thinking`` (newer Ollama) or a
      sibling ``reasoning`` field carries the user-visible text.
    * Tool-calling — ``message.tool_calls`` is a list of
      ``{function: {name, arguments}}`` dicts that we normalise into
      :class:`ToolCall` instances.
    """
    message = body.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        for fallback_key in ("thinking", "reasoning", "reasoning_content"):
            value = message.get(fallback_key) or body.get(fallback_key)
            if isinstance(value, str) and value.strip():
                content = value.strip()
                break

    raw_tool_calls = message.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for idx, call in enumerate(raw_tool_calls):
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name") or ""
        arguments = fn.get("arguments") or call.get("arguments") or {}
        tool_calls.append(
            ToolCall(
                tool_name=name,
                tool_id=str(call.get("id") or f"call_{idx}"),
                parameters=dict(arguments) if isinstance(arguments, dict) else {"raw": arguments},
            )
        )

    usage: dict[str, int] = {}
    if "prompt_eval_count" in body:
        usage["prompt_tokens"] = int(body["prompt_eval_count"])
    if "eval_count" in body:
        usage["completion_tokens"] = int(body["eval_count"])
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    stop_reason = body.get("done_reason") or ("stop" if body.get("done") else None)

    return ChatResult(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=stop_reason,
        raw=body,
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
