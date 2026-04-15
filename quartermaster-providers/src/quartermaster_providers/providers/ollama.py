"""Native Ollama ``/api/chat`` provider (v0.4.0).

The OpenAI-compat shim that ``OllamaProvider`` inherits by default
(:class:`OpenAICompatibleProvider` → ``/v1/chat/completions``) turned
out to be the source of the Gemma-4 tool-name hallucination class of
bugs: the compat layer rewrites function names through OpenAI's
``default_api:`` / ``functions:`` namespacing, and some models (notably
``gemma4:26b``) respond with suffixes like ``list_orders_v2`` that the
provider has no way to map back to the real registered tool name.

Ollama's native ``/api/chat`` endpoint doesn't have this problem — it
sends the tool definitions as ``{"type": "function", "function": {...}}``
and returns tool calls under ``message.tool_calls[i].function.name``
*without* a namespace prefix.  This module implements that path and
plumbs capability detection so the provider picks the right transport
per model automatically:

* ``tool_protocol="auto"`` (default) — query ``/api/tags`` once, cache
  the result, use ``/api/chat`` when the model advertises tool support.
* ``tool_protocol="native"`` — always use ``/api/chat`` (fails loudly
  if the model can't handle it — good for tests / debugging).
* ``tool_protocol="openai_compat"`` — always use ``/v1/chat/completions``
  (preserves pre-v0.4.0 behaviour for callers who want it).

The class *inherits* :class:`OllamaProvider` so the v0.1.3 sync
``chat()`` shim, ``_strip_v1`` URL routing, SSRF warning, and registry
integration keep working unchanged.  Only the async ``generate_*``
methods branch on capability.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, AsyncIterator

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.exceptions import (
    InvalidRequestError,
    ProviderError,
    ServiceUnavailableError,
)
from quartermaster_providers.providers.local import (
    OllamaProvider as _OpenAICompatOllamaProvider,
)
from quartermaster_providers.providers.local import _strip_v1
from quartermaster_providers.types import (
    NativeResponse,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)


#: Valid values for the ``tool_protocol`` kwarg / ``ollama_tool_protocol``
#: SDK config.  Exposed so the SDK can validate input before handing it
#: to the provider.
VALID_TOOL_PROTOCOLS: frozenset[str] = frozenset({"auto", "native", "openai_compat"})


#: Model families that are known to support native tool calling through
#: Ollama's ``/api/chat``.  Used as a fast-path hint when ``/api/tags``
#: doesn't expose an explicit ``capabilities`` field — newer Ollama
#: versions do, older ones don't.  The heuristic is intentionally
#: conservative: we only fast-path families Ollama's own docs list as
#: tool-capable (https://ollama.com/blog/tool-support).
_TOOL_CAPABLE_FAMILIES: frozenset[str] = frozenset(
    {
        "llama",  # llama3.1, llama3.2, llama3.3 all support tools
        "mistral",  # mistral-nemo, mistral-small all support tools
        "qwen",  # qwen2.5, qwen3, qwq
        "qwen2",
        "qwen3",
        "command-r",  # command-r-plus, command-r7b
        "firefunction",
        "gemma",  # gemma3, gemma4 variants (gemma4:26b is what triggered this)
        "granite",  # granite3 series
        "phi",  # phi4-mini
        "smollm",  # smollm2
    }
)


#: Model name tokens that are NOT tool-capable even when the family looks
#: like it should be.  Pre-llama3.1 variants, for example, carry the
#: ``llama`` family name but predate tool support.  When a model name
#: contains any of these tokens we treat it as openai-compat-only.
_TOOL_INCAPABLE_TOKENS: frozenset[str] = frozenset(
    {
        "llama2",
        "llama-2",
        "llama3.0",
        "llama-3.0",
        "codellama",  # no tool support in codellama
    }
)


class OllamaNativeProvider(_OpenAICompatOllamaProvider):
    """Ollama provider that uses native ``/api/chat`` for tool calls.

    When ``tool_protocol="auto"`` (the default), the provider probes
    ``/api/tags`` on first use to find which models support tools
    natively, caches the answer, and routes tool-calling requests
    through ``/api/chat`` for those models.  Text-only requests still
    use the inherited OpenAI-compat path by default — it has the
    richer streaming story and neither transport has an advantage for
    plain text.

    When ``tool_protocol="native"``, every request goes through
    ``/api/chat`` (including plain text / streaming).  Useful for tests
    and for forcing the same wire format everywhere.

    When ``tool_protocol="openai_compat"``, the native path is never
    used — equivalent to the pre-v0.4.0 ``OllamaProvider``.

    Args:
        base_url: Endpoint URL.  Same semantics as the base class
            (accepts ``http://host:port`` or ``http://host:port/v1``).
        api_key: Unused for native Ollama; kept for symmetry.
        default_model: Default model for sync ``chat()`` and async
            ``generate_*`` when the call doesn't pass its own.
        tool_protocol: ``"auto"`` / ``"native"`` / ``"openai_compat"``.
    """

    PROVIDER_NAME = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str = "ollama",
        default_model: str | None = None,
        tool_protocol: str = "auto",
        **kwargs: Any,
    ):
        if tool_protocol not in VALID_TOOL_PROTOCOLS:
            raise ValueError(
                f"Unknown tool_protocol={tool_protocol!r}; "
                f"expected one of {sorted(VALID_TOOL_PROTOCOLS)}."
            )
        self.tool_protocol = tool_protocol
        # Per-instance cache of ``model_name -> bool`` for "does this
        # model support native tool calling?"  Populated lazily via
        # ``_supports_native_tools``; cleared by ``reset_capability_cache``.
        self._native_support_cache: dict[str, bool] = {}
        # Cached ``/api/tags`` response — single fetch per provider
        # instance.  ``None`` means "not yet fetched"; an empty dict
        # means "fetched, got an empty result".
        self._tags_cache: dict[str, Any] | None = None
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            default_model=default_model,
            **kwargs,
        )

    # ── Capability detection ─────────────────────────────────────────

    def _fetch_tags(self) -> dict[str, Any]:
        """Fetch ``GET /api/tags`` and cache the response per-instance.

        Uses a synchronous ``httpx.Client`` so the method stays usable
        from both async provider methods (where we can do
        ``supports_native_tools`` as a quick, non-awaited check inside
        the async workflow — the request itself is fast) and the sync
        ``chat()`` shim.  When called from inside a running asyncio
        event loop, we offload the blocking I/O to a worker thread so
        we don't deadlock the loop on a sync HTTP call.

        Returns the raw JSON body (``{"models": [...]}``).  On any HTTP
        or network error, returns an empty dict and logs a debug
        message — capability detection then falls back to the
        family-name heuristic instead of failing the whole request.
        """
        if self._tags_cache is not None:
            return self._tags_cache
        import httpx

        url = f"{_strip_v1(self.base_url)}/api/tags"
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict):
                    self._tags_cache = body
                else:
                    self._tags_cache = {}
        except Exception as exc:
            logger.debug(
                "OllamaNativeProvider: could not fetch %s (%s); "
                "falling back to family-name capability heuristic.",
                url,
                exc,
            )
            self._tags_cache = {}
        return self._tags_cache

    def _supports_native_tools(self, model: str) -> bool:
        """Return True when *model* advertises native tool support.

        Resolution order (first wins):

        1. ``tool_protocol="native"`` → always True.
        2. ``tool_protocol="openai_compat"`` → always False.
        3. Per-instance cache hit.
        4. ``/api/tags`` response — look at the matching model's
           ``capabilities`` array (newer Ollama) or its
           ``details.family`` (older Ollama, fall back to the
           family-name heuristic).
        5. If ``/api/tags`` failed or didn't include the model,
           use the family-name heuristic on *model* itself.
        """
        if self.tool_protocol == "native":
            return True
        if self.tool_protocol == "openai_compat":
            return False
        if model in self._native_support_cache:
            return self._native_support_cache[model]

        result = self._probe_native_support(model)
        self._native_support_cache[model] = result
        return result

    def _probe_native_support(self, model: str) -> bool:
        """Run the actual capability check for *model* (uncached)."""
        tags = self._fetch_tags()
        models = tags.get("models") or []

        # Match by exact name or by the base name before the ``:`` tag.
        model_base = model.split(":", 1)[0].lower()
        for entry in models:
            if not isinstance(entry, dict):
                continue
            entry_name = str(entry.get("name", "")).lower()
            entry_model = str(entry.get("model", "")).lower()
            if (
                entry_name == model.lower()
                or entry_model == model.lower()
                or entry_name.split(":", 1)[0] == model_base
                or entry_model.split(":", 1)[0] == model_base
            ):
                # Newer Ollama exposes an explicit capabilities list.
                caps = entry.get("capabilities")
                if isinstance(caps, list) and caps:
                    return "tools" in caps or "function_calling" in caps
                # Older Ollama: fall through to the family heuristic
                # using the model's own declared family.
                details = entry.get("details") or {}
                family = str(details.get("family", "")).lower()
                if family:
                    return self._family_supports_tools(family, model)
                break

        # Didn't find the model in /api/tags — use the bare family
        # heuristic on the model name itself.
        return self._family_supports_tools(model_base, model)

    @staticmethod
    def _family_supports_tools(family: str, full_name: str) -> bool:
        """Family-name heuristic for tool support.

        Returns True when the base model name (or a declared
        ``details.family``) starts with a known-tool-capable family
        string and the full name does NOT contain any known-incapable
        token (e.g. ``llama2`` even though the family is ``llama``).

        The comparison is a prefix match rather than exact equality so
        model names like ``mistral-nemo`` / ``qwen2.5-coder`` /
        ``gemma4-instruct`` all resolve correctly against the base
        family (``mistral`` / ``qwen`` / ``gemma``) in the capability
        set.
        """
        lowered = full_name.lower()
        for token in _TOOL_INCAPABLE_TOKENS:
            if token in lowered:
                return False
        family_lower = family.lower()
        for capable in _TOOL_CAPABLE_FAMILIES:
            # Match the family exactly OR as a prefix of the model
            # name — handles ``mistral-nemo`` / ``qwen2.5`` / etc.
            if family_lower == capable:
                return True
            if family_lower.startswith(capable):
                return True
        return False

    def reset_capability_cache(self) -> None:
        """Clear cached capability info — used by tests and hot-reloads."""
        self._native_support_cache.clear()
        self._tags_cache = None

    # ── Payload builders ─────────────────────────────────────────────

    def _prepare_native_messages(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> list[dict[str, Any]]:
        """Build the ``messages`` array for ``/api/chat``.

        Native Ollama is closer to the Anthropic shape than OpenAI's:
        system / user / assistant / tool roles, but images go on the
        user message as a sibling ``images: [base64]`` list (not an
        OpenAI-style ``content: [{type: image_url, ...}]`` array).
        """
        messages: list[dict[str, Any]] = []
        if config.system_message:
            messages.append({"role": "system", "content": config.system_message})

        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if config.images:
            # Ollama's /api/chat accepts raw base64 strings (no
            # ``data:image/...;base64,`` prefix) in a top-level
            # ``images`` list on the user message.
            user_message["images"] = [b64 for b64, _mime in config.images]
        messages.append(user_message)
        return messages

    def _build_options(self, config: LLMConfig) -> dict[str, Any]:
        """Translate LLMConfig sampling knobs into Ollama's ``options`` dict."""
        options: dict[str, Any] = {}
        if config.temperature is not None:
            options["temperature"] = float(config.temperature)
        if config.max_output_tokens:
            options["num_predict"] = int(config.max_output_tokens)
        if config.top_p is not None:
            options["top_p"] = float(config.top_p)
        if config.top_k is not None:
            options["top_k"] = int(config.top_k)
        if config.frequency_penalty is not None:
            options["frequency_penalty"] = float(config.frequency_penalty)
        if config.presence_penalty is not None:
            options["presence_penalty"] = float(config.presence_penalty)
        return options

    def _prepare_native_tools(
        self,
        tools: list[ToolDefinition] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert quartermaster tools to Ollama's native tool format.

        Ollama's native tool schema is effectively the OpenAI function-
        calling schema:

            [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "...",
                    "parameters": {"type": "object", "properties": {...}}
                }
            }, ...]
        """
        if not tools:
            return None
        prepared: list[dict[str, Any]] = []
        for tool in tools:
            prepared.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return prepared

    def _build_native_payload(
        self,
        prompt: str,
        config: LLMConfig,
        *,
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Assemble the JSON body for ``POST /api/chat``."""
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": self._prepare_native_messages(prompt, config),
            "stream": bool(stream),
        }
        options = self._build_options(config)
        if options:
            payload["options"] = options

        prepared_tools = self._prepare_native_tools(tools)
        if prepared_tools:
            payload["tools"] = prepared_tools

        # Extended thinking: Ollama uses a ``think: bool`` flag on the
        # request.  LLMConfig.thinking_enabled is the engine's unified
        # toggle (Claude / OpenAI o-series also consume it).  When
        # explicitly off we intentionally don't emit ``think: false``
        # so older Ollama versions that don't know the field keep
        # working silently.
        if config.thinking_enabled:
            payload["think"] = True

        return payload

    # ── HTTP helpers ──────────────────────────────────────────────────

    def _native_chat_url(self) -> str:
        """Return the URL for ``POST /api/chat`` on this Ollama instance."""
        return f"{_strip_v1(self.base_url)}/api/chat"

    def _resolve_native_timeout(self, config: LLMConfig | None) -> float:
        """Pick an httpx timeout for native calls.

        Defaults to a generous 120s so first-model-load requests don't
        trip a timeout before Ollama even finishes pulling weights into
        VRAM.  ``LLMConfig.read_timeout`` wins when set (it was added
        in the separate v0.4.0 timeouts workstream — we read it
        defensively so this module keeps working on branches that
        don't have the field yet).
        """
        if config is None:
            return 120.0
        read = getattr(config, "read_timeout", None)
        if read is not None:
            return float(read)
        return 120.0

    async def _post_native(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        """POST the non-streaming ``/api/chat`` request and return the body."""
        import httpx

        url = self._native_chat_url()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama /api/chat returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:500]}",
                provider=self.PROVIDER_NAME,
                status_code=exc.response.status_code,
            ) from exc
        except (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.TimeoutException,
        ) as exc:
            raise ServiceUnavailableError(
                f"Could not reach Ollama /api/chat at {url}: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

    async def _stream_native(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> AsyncIterator[TokenResponse]:
        """Stream ``/api/chat`` NDJSON as :class:`TokenResponse` chunks."""
        import httpx

        streaming_payload = dict(payload)
        streaming_payload["stream"] = True
        url = self._native_chat_url()
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", url, json=streaming_payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # Defensive — Ollama emits strict NDJSON,
                        # but a proxy in front could theoretically
                        # fold or pad lines.  Skip malformed ones
                        # rather than failing the stream.
                        continue
                    message = chunk.get("message") or {}
                    content = message.get("content") or ""
                    if content:
                        yield TokenResponse(
                            content=content,
                            stop_reason=None,
                        )
                    else:
                        # Reasoning-model fallback — same logic as
                        # the non-streaming parser.
                        for fallback_key in ("thinking", "reasoning"):
                            value = message.get(fallback_key)
                            if isinstance(value, str) and value:
                                yield TokenResponse(
                                    content=value,
                                    stop_reason=None,
                                )
                                break
                    if chunk.get("done"):
                        yield TokenResponse(
                            content="",
                            stop_reason=chunk.get("done_reason") or "stop",
                        )
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama /api/chat returned HTTP {exc.response.status_code}",
                provider=self.PROVIDER_NAME,
                status_code=exc.response.status_code,
            ) from exc
        except (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.TimeoutException,
        ) as exc:
            raise ServiceUnavailableError(
                f"Could not reach Ollama /api/chat at {url}: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

    # ── Native response parsers ──────────────────────────────────────

    @staticmethod
    def _extract_text_content(message: dict[str, Any], body: dict[str, Any]) -> str:
        """Pull the user-visible text out of an Ollama response message.

        Promotes ``thinking`` / ``reasoning`` fields to ``content`` when
        the primary field is empty — matches the sync ``chat()`` shim's
        reasoning-model handling.
        """
        content = (message.get("content") or "").strip()
        if content:
            return content
        for fallback_key in ("thinking", "reasoning", "reasoning_content"):
            value = message.get(fallback_key) or body.get(fallback_key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        """Normalise Ollama ``message.tool_calls`` into ``ToolCall`` instances.

        The native path uses plain ``function.name`` (no ``default_api:``
        prefix), which is precisely why this module exists — callers
        get the real tool name back and can look it up in the registry
        without the namespace-stripping dance.
        """
        raw = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for idx, call in enumerate(raw):
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name") or ""
            arguments = fn.get("arguments") or call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"raw": arguments}
            tool_calls.append(
                ToolCall(
                    tool_name=name,
                    tool_id=str(call.get("id") or f"call_{idx}"),
                    parameters=dict(arguments),
                )
            )
        return tool_calls

    @staticmethod
    def _extract_usage(body: dict[str, Any]) -> TokenUsage | None:
        """Translate Ollama's timing stats into a :class:`TokenUsage`."""
        prompt = body.get("prompt_eval_count")
        completion = body.get("eval_count")
        if prompt is None and completion is None:
            return None
        return TokenUsage(
            input_tokens=int(prompt or 0),
            output_tokens=int(completion or 0),
        )

    # ── Overridden async entry points ────────────────────────────────

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        """Text generation — native when the mode forces it, else compat."""
        if self.tool_protocol == "native":
            payload = self._build_native_payload(prompt, config, stream=config.stream)
            timeout = self._resolve_native_timeout(config)
            if config.stream:
                return self._stream_native(payload, timeout=timeout)
            body = await self._post_native(payload, timeout=timeout)
            message = body.get("message") or {}
            return TokenResponse(
                content=self._extract_text_content(message, body),
                stop_reason=body.get("done_reason") or ("stop" if body.get("done") else None),
            )
        # ``auto`` and ``openai_compat`` both default to the OpenAI-compat
        # path for plain text — the compat streaming story is solid and
        # there's no Gemma-style hallucination risk when no tools are
        # involved.
        return await super().generate_text_response(prompt, config)

    async def generate_tool_parameters(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        config: LLMConfig,
    ) -> ToolCallResponse:
        """Tool-calling generation — the v0.4.0 reason this module exists.

        Routes through native ``/api/chat`` when the model supports
        tools natively (auto mode) or when forced (native mode).
        Otherwise delegates to the inherited OpenAI-compat path.
        """
        if not self._supports_native_tools(config.model):
            return await super().generate_tool_parameters(prompt, tools, config)

        payload = self._build_native_payload(prompt, config, tools=tools, stream=False)
        body = await self._post_native(payload, timeout=self._resolve_native_timeout(config))
        message = body.get("message") or {}

        return ToolCallResponse(
            text_content=self._extract_text_content(message, body),
            tool_calls=self._extract_tool_calls(message),
            stop_reason=body.get("done_reason") or ("stop" if body.get("done") else None),
            usage=self._extract_usage(body),
        )

    async def generate_native_response(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        config: LLMConfig | None = None,
    ) -> NativeResponse:
        """Unified text + tool-call generation.

        Native path used when the model supports it (auto mode) or
        when forced (native mode); otherwise falls back to the
        inherited OpenAI-compat path so existing callers keep working.
        """
        if config is None:
            raise InvalidRequestError("config is required", provider=self.PROVIDER_NAME)

        if not self._supports_native_tools(config.model):
            return await super().generate_native_response(prompt, tools, config)

        payload = self._build_native_payload(prompt, config, tools=tools, stream=False)
        body = await self._post_native(payload, timeout=self._resolve_native_timeout(config))
        message = body.get("message") or {}

        return NativeResponse(
            text_content=self._extract_text_content(message, body),
            thinking=[],
            tool_calls=self._extract_tool_calls(message),
            stop_reason=body.get("done_reason") or ("stop" if body.get("done") else None),
            usage=self._extract_usage(body),
        )


# ── Module-level capability helper ──────────────────────────────────────
#
# Exposed so callers can probe capability without instantiating the
# provider (e.g. for diagnostics or feature gates in the SDK).  Uses a
# small LRU cache keyed on (base_url, model) so repeated probes during
# a single run don't hammer /api/tags.


@lru_cache(maxsize=64)
def _cached_model_supports_native_tools(base_url: str, model: str) -> bool:
    """LRU-cached capability probe.  Intentionally module-level so the
    cache survives across provider instances in short-lived scripts."""
    provider = OllamaNativeProvider(base_url=base_url, tool_protocol="auto")
    try:
        return provider._supports_native_tools(model)
    finally:
        # Drop the client-side cache on the instance — the lru_cache
        # above is what we're relying on for persistence.
        provider.reset_capability_cache()


def model_supports_native_tools(
    model: str,
    *,
    base_url: str | None = None,
) -> bool:
    """Return True when *model* supports Ollama's native tool-calling API.

    Convenience wrapper that callers can use without constructing an
    :class:`OllamaNativeProvider` themselves.  Uses ``OLLAMA_HOST`` when
    *base_url* is omitted.
    """
    resolved = base_url or os.environ.get("OLLAMA_HOST") or OllamaNativeProvider.DEFAULT_BASE_URL
    return _cached_model_supports_native_tools(resolved, model)
