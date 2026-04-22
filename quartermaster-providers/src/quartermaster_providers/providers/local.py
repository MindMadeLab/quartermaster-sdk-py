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


# Same set the engine's ``_build_llm_config`` validates against — kept
# in sync by name (defining it here too rather than importing across
# the engine→providers boundary, which would invert the dep direction).
_VALID_THINKING_LEVELS: frozenset[str] = frozenset({"off", "low", "medium", "high"})


# Hostnames / IPs that point at cloud-metadata or link-local services —
# legitimate Ollama deployments never live here.  Operators *could* set
# these intentionally for niche tunneling setups, so we warn rather than
# block, but the warning makes it visible in logs when a misconfiguration
# (or, in multi-tenant settings, a hostile end-user-supplied base_url)
# would point an HTTP request at IMDS / link-local space.
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
    """Log a warning when *base_url* points at a known SSRF-magnet host.

    Operators set ``base_url`` at registration time, so this is at most
    a misconfiguration alarm in single-tenant deployments.  In multi-
    tenant setups where end users can supply their own provider URL it
    becomes a soft SSRF gate — the warning lands in operator logs even
    if the hostile request still goes through.
    """
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


def _strip_v1(base_url: str) -> str:
    """Drop the ``/v1`` segment that ``_normalize_openai_compat_url`` adds.

    The OpenAI-compat path lives at ``/v1/chat/completions`` but the
    native Ollama API lives at ``/api/chat``.  ``OllamaProvider`` stores
    its base URL in the ``/v1`` form (so the openai SDK works), so the
    sync ``chat()`` shim has to peel ``/v1`` back off before talking to
    ``/api/chat``.

    Only strips when ``/v1`` sits directly under the host root —
    ``http://host:port/v1``.  A URL like ``http://gateway/api/v1`` is
    likely a corporate proxy with its own path prefix (where stripping
    would silently produce ``http://gateway/api`` and route ``/api/chat``
    requests to ``/api/api/chat``).  In that case we leave the URL alone.

    When stripping, the result is reconstructed from scheme + host (+
    port) only — userinfo, query, and fragment from the input are
    intentionally dropped.  This blunts the attack class where a
    base_url like ``http://user:pass@host/v1`` would otherwise leak the
    credentials into the outbound ``/api/chat`` request, and where a
    ``http://host/v1?injected=...`` query string would silently mangle
    the final URL.
    """
    if not base_url:
        return base_url
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path == "/v1":
        # Reconstruct from trusted components only — drop userinfo,
        # query, and fragment so they can't ride along into the
        # request URL.  hostname + port is what we actually need.
        host = parsed.hostname or ""
        if parsed.port is not None:
            netloc = f"{host}:{parsed.port}"
        else:
            netloc = host
        return urlunparse((parsed.scheme, netloc, "", "", "", ""))
    # Anything else (deeper path, or no /v1 at all) is left untouched.
    return base_url.rstrip("/")


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
        default_model: str | None = None,
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ):
        resolved = base_url or os.environ.get("OLLAMA_HOST") or self.DEFAULT_BASE_URL
        _warn_if_suspicious_url(resolved)
        self._chat_default_model = default_model

        # v0.4.3: HTTP Basic Auth + custom headers for Ollama behind
        # reverse proxies (nginx, Caddy, Traefik). ``auth`` is a
        # (username, password) tuple; ``headers`` are extra HTTP headers
        # injected into every request (both the OpenAI SDK path and the
        # native httpx path).
        #
        # Resolution: ``auth`` kwarg > OLLAMA_USER + OLLAMA_PASS env vars.
        # ``headers`` kwarg > OLLAMA_HEADERS env var (comma-separated
        # ``Key:Value`` pairs, e.g. ``X-Token:abc,X-Org:sorex``).
        self._http_auth: tuple[str, str] | None = auth or _env_auth()
        self._extra_headers: dict[str, str] = dict(headers or _env_headers())

        # Pick auth_method for the OpenAI SDK async path.
        auth_method = "none"
        auth_credentials: tuple[str, str] | None = None
        if self._http_auth:
            auth_method = "basic"
            auth_credentials = self._http_auth

        super().__init__(
            base_url=_normalize_openai_compat_url(resolved),
            api_key=api_key,
            auth_method=auth_method,
            auth_credentials=auth_credentials,
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
    #
    # NOTE for future maintainers: ``chat()`` is INTENTIONALLY synchronous
    # alongside the inherited async ``generate_*`` methods.  Do not add a
    # ``chat_async()`` cousin — if you need async on this provider, call
    # ``generate_native_response`` (the OpenAI-compat path).  Two flavours
    # of the same call are easier to maintain than three.

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
            model: Model id (e.g. ``"gemma4:26b"``).  When omitted, the
                provider's registration-time ``default_model`` is used
                (set via ``register_local("ollama", default_model=...)``).
                If neither is supplied the call raises ``ProviderError``.
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
                or ``None`` to leave it to the model.  Anything else
                logs a warning and is treated as ``off``.  Forwarded as
                Ollama's ``think`` boolean.
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
            # Ollama's native parameter name is ``think`` (bool); we
            # accept the higher-level enum so callers don't have to
            # know which version of Ollama they're talking to.  Validate
            # against the same set the engine's ``_build_llm_config``
            # uses so a typo here can't silently flip ``think`` on.
            if thinking_level not in _VALID_THINKING_LEVELS:
                logger.warning(
                    "Unknown thinking_level=%r for OllamaProvider.chat() "
                    "(expected one of %s); falling back to 'off'.",
                    thinking_level,
                    sorted(_VALID_THINKING_LEVELS),
                )
                payload["think"] = False
            else:
                payload["think"] = thinking_level != "off"

        url = f"{_strip_v1(self.base_url)}/api/chat"

        try:
            client_kwargs: dict[str, Any] = {"timeout": timeout}
            if self._http_auth:
                client_kwargs["auth"] = self._http_auth
            if self._extra_headers:
                client_kwargs["headers"] = self._extra_headers
            with httpx.Client(**client_kwargs) as client:
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
        except (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.TimeoutException,
        ) as exc:
            # Network unreachable, write failed, or any flavour of
            # timeout — connect/read/write/pool — all mean the request
            # didn't get a usable answer from Ollama.  ``TimeoutException``
            # is the umbrella for ``ConnectTimeout`` (server up but slow
            # to accept the socket — common during model load) plus
            # ``ReadTimeout`` and ``WriteTimeout``.  Catching only
            # ``ReadTimeout`` would let ``ConnectTimeout`` bubble out as
            # a generic ``ProviderError`` and confuse callers about
            # whether Ollama was reachable at all.
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
        """Default model for the sync ``chat()`` shim.

        Set by :meth:`__init__` from the ``default_model`` constructor
        kwarg, which :meth:`ProviderRegistry.register_local` forwards
        through whenever ``register_local("ollama", default_model=...)``
        was used.  Returns ``None`` for instances constructed by hand
        without a default — in which case ``chat()`` will require an
        explicit ``model=`` per call.
        """
        return self._chat_default_model


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
#
# v0.4.0: the shorthand ``"ollama"`` now resolves to
# :class:`OllamaNativeProvider` rather than the compat-only
# ``OllamaProvider`` above.  The native subclass still inherits all of
# the OpenAI-compat plumbing — so the sync ``chat()`` shim, SSRF
# warning, URL normalisation, and default-model threading keep
# working — but it *also* routes tool-calling requests through
# Ollama's native ``/api/chat`` endpoint when the model supports it.
# That kills the Gemma-4 ``list_orders_v2`` / ``default_api:`` tool-
# name hallucination class of bugs the compat path was prone to.
#
# We import ``OllamaNativeProvider`` inside a factory-style lookup
# (via ``__getattr__`` on the module) to keep the import cycle sane:
# ``ollama.py`` imports ``local.OllamaProvider`` as its base class, so
# ``local.py`` can't unconditionally import ``ollama.py`` at module
# load time.  The indirection is cheap — it only fires the first time
# someone actually calls ``register_local("ollama")``.


def _lookup_ollama_provider_class() -> type[OpenAICompatibleProvider]:
    """Return the default Ollama provider class (lazy import).

    Lazy so ``ollama.py`` (which inherits from this module's
    :class:`OllamaProvider`) can import cleanly without a circular
    dependency.  Callers who want the raw OpenAI-compat behaviour for
    some reason can still construct ``OllamaProvider`` directly from
    this module.
    """
    from quartermaster_providers.providers.ollama import OllamaNativeProvider

    return OllamaNativeProvider


class _LocalProvidersLookup(dict):
    """dict that lazily resolves ``"ollama"`` to :class:`OllamaNativeProvider`.

    Mirrors ``dict.__getitem__`` / ``dict.get`` semantics so existing
    callsites (``LOCAL_PROVIDERS[engine]``) keep working unchanged,
    but defers the ``ollama.py`` import until the key is actually
    accessed.  This avoids the ``local.py`` → ``ollama.py`` → ``local.py``
    circular import that a top-level ``from .ollama import ...`` would
    produce (``ollama.py`` subclasses ``OllamaProvider``).
    """

    def __getitem__(self, key: str) -> type[OpenAICompatibleProvider]:
        if key == "ollama":
            return _lookup_ollama_provider_class()
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "ollama":
            return _lookup_ollama_provider_class()
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if key == "ollama":
            return True
        return super().__contains__(key)

    def values(self):  # type: ignore[override]
        """Include the native Ollama class when callers iterate ``.values()``.

        Regression guard: the test ``test_all_are_openai_compatible``
        iterates ``LOCAL_PROVIDERS.values()`` and checks the subclass
        invariant — without this override the native class would be
        missing from the iteration (since it only exists via
        ``__getitem__``) and the check would fall through silently.
        """
        items = dict(self)
        items["ollama"] = _lookup_ollama_provider_class()
        return items.values()


LOCAL_PROVIDERS: dict[str, type[OpenAICompatibleProvider]] = _LocalProvidersLookup(
    {
        # NOTE: ``"ollama"`` resolves lazily via __getitem__/__contains__
        # above; the sentinel here keeps ``keys()`` / iteration sane.
        "ollama": OllamaProvider,
        "vllm": VLLMProvider,
        "lm-studio": LMStudioProvider,
        "tgi": TGIProvider,
        "localai": LocalAIProvider,
        "llama-cpp": LlamaCppProvider,
    }
)
"""Maps shorthand names to provider classes for :meth:`ProviderRegistry.register_local`.

Note: ``"ollama"`` resolves to :class:`OllamaNativeProvider` (the
v0.4.0 native ``/api/chat`` subclass) via a lazy lookup to keep the
module import graph acyclic.  Callers who specifically want the
compat-only behaviour can import :class:`OllamaProvider` directly.
"""
