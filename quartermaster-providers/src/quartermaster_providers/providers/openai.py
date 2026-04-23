"""OpenAI LLM provider implementation (GPT-4o, GPT-4, o-series, Whisper).

Implements AbstractLLMProvider for OpenAI's API, supporting text generation,
streaming, tool calling, structured output, vision, and audio transcription.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, NoReturn, cast

from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.config import LLMConfig
from quartermaster_providers.exceptions import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)
from quartermaster_providers.types import (
    NativeResponse,
    StructuredResponse,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)

IMAGE_TOKEN_ESTIMATE = 1000

# Known OpenAI models
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
]

# Pricing per 1K tokens (USD) as of 2025
OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
}


def _is_o_series(model: str) -> bool:
    """Check if a model is an o-series reasoning model."""
    return model.startswith("o1") or model.startswith("o3")


#: Regex pairs that match the text-form tool-call markers emitted by
#: mis-configured vLLM / Ollama servers (server was started without a
#: ``--tool-call-parser`` flag, so the chat template's literal
#: ``<|tool_call|>`` sentinels leak into ``message.content`` instead of
#: being converted into structured ``tool_calls`` by the server). Each
#: entry is ``(open_marker, close_marker)``. Order-matters: we try the
#: most specific marker set first.
_TEXT_TOOL_CALL_MARKERS: tuple[tuple[str, str], ...] = (
    # Gemma-4's actual production form — asymmetric pipes. Open marker
    # is ``<|tool_call>`` (pipe AFTER the ``<``), close marker is
    # ``<tool_call|>`` (pipe BEFORE the ``>``). A production reproducer:
    #   <|tool_call>call:list_orders(status='odprto')<tool_call|>
    # Tried FIRST because it's more specific — the symmetric form
    # below would accidentally match half of the asymmetric markers.
    ("<|tool_call>", "<tool_call|>"),
    # Symmetric Gemma variant — appears in some fine-tunes and in the
    # upstream vLLM chat template when the stop tokens are collapsed.
    ("<|tool_call|>", "<|tool_call|>"),
    # Qwen / some Llama fine-tunes
    ("<tool_call>", "</tool_call>"),
    # Fireworks / some custom templates
    ("[TOOL_CALLS]", "[/TOOL_CALLS]"),
)

_TOOL_CALL_REGEXES = None


def _compiled_tool_call_regexes():
    """Compile once, cache. Keeps import time clean."""
    global _TOOL_CALL_REGEXES
    if _TOOL_CALL_REGEXES is None:
        import re

        _TOOL_CALL_REGEXES = tuple(
            re.compile(
                re.escape(open_m) + r"\s*(.*?)\s*" + re.escape(close_m),
                re.DOTALL,
            )
            for open_m, close_m in _TEXT_TOOL_CALL_MARKERS
        )
    return _TOOL_CALL_REGEXES


def _parse_text_form_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Salvage tool calls from a mis-configured server's text-form output.

    Returns ``(tool_calls, residual_text)`` — ``tool_calls`` is whatever we
    could recover from the marker-delimited blocks; ``residual_text`` is
    ``content`` with the marker blocks stripped out (so the agent's visible
    text answer doesn't carry the literal ``<|tool_call|>...`` junk).

    Each marker-delimited block is parsed as JSON shaped like one of:

        {"name": "foo", "arguments": {...}}
        {"tool_name": "foo", "parameters": {...}}
        {"function": {"name": "foo", "arguments": "{json}"}}

    Blocks that don't parse are skipped silently (ops can still see the
    raw content in ``message.content``; we don't want to raise from a
    best-effort salvage).

    This is a v0.6.0 defence-in-depth: servers launched with
    ``--tool-call-parser <flavour>`` already emit structured
    ``tool_calls`` and never trip this path. But users who forget the
    flag — or who run against an older vLLM — used to silently lose tool
    calls (the agent saw text and exited thinking it had a final answer).
    """
    if not content:
        return [], content
    residual = content
    recovered: list[ToolCall] = []
    for regex in _compiled_tool_call_regexes():
        for idx, match in enumerate(regex.finditer(content)):
            payload = match.group(1).strip()
            parsed = _coerce_text_tool_call_payload(payload)
            if parsed is None:
                continue
            name, arguments = parsed
            recovered.append(
                ToolCall(
                    tool_name=name,
                    tool_id=f"call_text_{idx}",
                    parameters=arguments,
                )
            )
        if recovered:
            residual = regex.sub("", residual)
            # Stop at the first marker flavour that produced hits.
            # Servers usually emit one flavour only, and strip-then-re-
            # scanning with a different regex would double-consume text.
            break
    return recovered, residual.strip()


def _coerce_text_tool_call_payload(payload: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one tool-call block into ``(name, arguments_dict)``.

    Two payload flavours supported:

    1. **JSON** — ``{"name": ..., "arguments": {...}}`` /
       ``{"tool_name": ..., "parameters": {...}}`` /
       ``{"function": {"name": ..., "arguments": {...}}}``, plus
       double-encoded arguments strings.
    2. **Python-call syntax** — ``call:list_orders(status='active')``.
       Gemma-4 emits this form inside ``<|tool_call>...<tool_call|>``
       blocks when the server lacks ``--tool-call-parser gemma4``.
       Keyword args supported (``key='value'`` / ``key=42`` / ``key=None``
       / ``key=True``). Positional-only calls are NOT supported —
       downstream tool registries expect keyword arguments; a
       positional call would need per-tool signature introspection
       we don't have at this layer.

    Returns ``None`` when neither form parses.
    """
    payload = payload.strip()
    if not payload:
        return None

    # JSON path first — it's strict, so if it doesn't parse we know
    # to try the Python-call form. Parsing-unit has no ambiguity.
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return _coerce_json_payload(obj)
    except json.JSONDecodeError:
        pass

    # Python-call path — ``call:name(key='val', key2=42, ...)``.
    return _coerce_python_call_payload(payload)


def _coerce_json_payload(obj: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """JSON-shape extraction — factored out so the JSON and
    Python-call paths stay easy to read independently."""
    fn = obj.get("function") if "function" in obj else obj
    if not isinstance(fn, dict):
        return None

    name = fn.get("name") or fn.get("tool_name")
    if not isinstance(name, str) or not name:
        return None

    args = fn.get("arguments")
    if args is None:
        args = fn.get("parameters", {})
    if isinstance(args, str):
        # Some templates double-encode JSON as a string.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    if not isinstance(args, dict):
        args = {"raw": args}
    return name, args


#: Matches ``call:foo(args...)`` and ``foo(args...)`` with optional
#: ``call:`` prefix. Captures the function name and the raw argument
#: string. ``DOTALL`` so arg strings spanning newlines (rare, but
#: Gemma-4 occasionally wraps them) still match.
_PYTHON_CALL_RE = None


def _python_call_regex():
    global _PYTHON_CALL_RE
    if _PYTHON_CALL_RE is None:
        import re

        _PYTHON_CALL_RE = re.compile(
            r"""
            (?:call\s*:\s*)?           # optional ``call:`` prefix
            ([A-Za-z_][A-Za-z0-9_]*)   # tool name — Python identifier
            \s*\(                      # opening paren
            (.*?)                      # arg body (lazy, stops at the matching close)
            \)\s*$                     # closing paren + optional trailing whitespace
            """,
            re.DOTALL | re.VERBOSE,
        )
    return _PYTHON_CALL_RE


def _coerce_python_call_payload(payload: str) -> tuple[str, dict[str, Any]] | None:
    """Parse ``call:list_orders(status='active', limit=5)`` into
    ``("list_orders", {"status": "active", "limit": 5})``.

    Uses :mod:`ast` to evaluate argument values so we get real Python
    scalars (ints, floats, bools, ``None``, strings, lists, dicts).
    ``ast.literal_eval`` rejects arbitrary expressions — no
    ``eval()`` risk, no ``__import__``-style escapes.
    """
    import ast

    match = _python_call_regex().fullmatch(payload)
    if match is None:
        return None
    name, raw_args = match.group(1), match.group(2).strip()
    if not name:
        return None

    if not raw_args:
        # ``call:list_orders()`` — no arguments.
        return name, {}

    # Wrap the arg body in a synthetic call so ast can parse the
    # keyword arguments for us — this is strictly-speaking easier
    # than splitting on commas (which gets ugly inside nested dicts,
    # quoted commas, etc.).
    try:
        expr = ast.parse(f"_({raw_args})", mode="eval")
    except SyntaxError:
        return None

    call = expr.body
    if not isinstance(call, ast.Call):
        return None

    args: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            # ``**kwargs`` splat — nothing useful to extract; skip.
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            # Value wasn't a literal (e.g. a bare identifier the model
            # invented); stash the source form so the tool registry can
            # at least log it meaningfully.
            args[kw.arg] = ast.unparse(kw.value)

    # Positional args: we expose them as ``__positional__`` so downstream
    # tools can see them if they really want to, but most tools in
    # quartermaster expect kwargs only. Left in as a diagnostic, not a
    # promise.
    if call.args:
        args["__positional__"] = [
            (ast.literal_eval(a) if _is_literal(a) else ast.unparse(a)) for a in call.args
        ]

    return name, args


def _is_literal(node: Any) -> bool:
    """``True`` when ``ast.literal_eval`` would accept *node*."""
    import ast

    try:
        ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return False
    return True


async def _aclose_stream(stream: Any) -> None:
    """Close an openai streaming response defensively.

    openai's ``AsyncStream`` exposes either ``close()`` (a coroutine),
    ``aclose()`` (a coroutine), or a ``.response`` attribute with an
    ``aclose()`` coroutine. Older versions only have one or the other.
    We try each option; any exception during close is swallowed —
    we're in a finally path and must not obscure the original error.
    """
    for attr in ("close", "aclose"):
        method = getattr(stream, attr, None)
        if callable(method):
            try:
                result = method()
                if hasattr(result, "__await__"):
                    await result
                return
            except Exception:
                pass
    response = getattr(stream, "response", None)
    aclose = getattr(response, "aclose", None) if response is not None else None
    if callable(aclose):
        try:
            await aclose()
        except Exception:
            pass


def _extract_reasoning_text(obj: Any) -> str:
    """Pull reasoning text out of a non-standard OpenAI-compatible response.

    Some OpenAI-compatible servers (notably Ollama proxies for reasoning
    models like ``gemma4:26b``) leave ``content`` empty and put the user-
    visible answer in a ``reasoning`` or ``reasoning_content`` field.  This
    helper checks both attribute access (Pydantic model) and dict access so
    it works for any shape the SDK returns.

    The fallback is intentionally narrow: only non-empty string values
    trigger it, so a vanilla OpenAI response with empty content + empty
    reasoning still returns ``""`` (preserving "model said nothing"
    semantics for content-filtered or zero-temperature edge cases).
    """
    if obj is None:
        return ""
    for key in ("reasoning_content", "reasoning"):
        value = getattr(obj, key, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class OpenAIProvider(AbstractLLMProvider):
    """OpenAI LLM provider for GPT-4o, GPT-4, o-series, and Whisper.

    Args:
        api_key: OpenAI API key.
        organization_id: Optional OpenAI organization ID.
        base_url: Optional custom API endpoint (for Azure, proxies, etc.).
    """

    PROVIDER_NAME = "openai"

    def __init__(
        self,
        api_key: str,
        organization_id: str | None = None,
        base_url: str | None = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai package required for OpenAIProvider. "
                "Install with: pip install quartermaster-providers[openai]"
            ) from e

        self.api_key = api_key
        self.organization_id = organization_id
        self.base_url = base_url
        # Legacy single-slot attribute. Kept for back-compat (some downstream
        # code reads it directly); always mirrors the client for the current
        # loop. The real cache is ``_clients_by_loop`` below.
        self._client = None
        # Per-loop client cache. Each ``openai.AsyncOpenAI`` carries an
        # httpx.AsyncClient whose connection pool and asyncio primitives
        # (Event/Lock) bind to the loop they are first awaited on. Reusing
        # that client from a different loop — e.g. when a @tool() body calls
        # qm.run() which spins its own asyncio.run() — wedges those primitives
        # and surfaces as "RuntimeError: <Event> is bound to a different event
        # loop" / "Event loop is closed" from httpcore on the next call from
        # the original loop. Keeping one client per live loop lets nested
        # qm.run() from inside a @tool() body coexist with the outer agent's
        # connection pool. ``id(loop)`` keys rather than the loop object
        # itself so we can GC-clean dead entries without holding refs.
        self._clients_by_loop: dict[int, tuple[Any, Any]] = {}

    def _get_client(self):
        import asyncio

        # Back-compat: if a caller (test or integrator) has directly
        # assigned ``provider._client = <fake>``, honour it and skip the
        # per-loop pool. We detect an external injection as: ``self._client``
        # is set to an object we haven't registered in ``_clients_by_loop``.
        if self._client is not None and not any(
            c is self._client for (_, c) in self._clients_by_loop.values()
        ):
            return self._client

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Prune entries whose loop has closed. ``id()`` can be recycled so
        # identity via the held ref is what we check here, not the key.
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

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization_id,
            base_url=self.base_url,
        )
        self._clients_by_loop[loop_key] = (current_loop, client)
        self._client = client
        return client

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """Translate OpenAI SDK exceptions to quartermaster-providers exceptions."""
        import openai

        if isinstance(e, openai.AuthenticationError):
            raise AuthenticationError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, openai.RateLimitError):
            raise RateLimitError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, openai.BadRequestError):
            msg = str(e).lower()
            if "context_length" in msg or "maximum context" in msg or "too many tokens" in msg:
                raise ContextLengthError(str(e), provider=self.PROVIDER_NAME) from e
            if "content_filter" in msg or "content_policy" in msg:
                raise ContentFilterError(str(e), provider=self.PROVIDER_NAME) from e
            raise InvalidRequestError(str(e), provider=self.PROVIDER_NAME) from e
        if isinstance(e, openai.APIStatusError):
            if e.status_code == 503:
                raise ServiceUnavailableError(str(e), provider=self.PROVIDER_NAME) from e
            raise ProviderError(
                str(e), provider=self.PROVIDER_NAME, status_code=e.status_code
            ) from e
        if isinstance(e, openai.APIConnectionError):
            raise ServiceUnavailableError(str(e), provider=self.PROVIDER_NAME) from e
        raise ProviderError(str(e), provider=self.PROVIDER_NAME) from e

    def _build_user_content(self, prompt: str, config: LLMConfig) -> str | list[dict[str, Any]]:
        """Build the user-turn ``content`` field for the Chat Completions API.

        Text-only requests keep the plain string shortcut so tokens /
        payloads stay unchanged for 99% of callsites. When the caller
        attached images via ``LLMConfig.images`` we emit the structured
        content-part list the OpenAI SDK expects: each image becomes an
        ``image_url`` part with a ``data:<mime>;base64,<data>`` URL, and
        the original text prompt comes last.
        """
        if not config.images:
            return prompt
        parts: list[dict[str, Any]] = []
        for b64_data, mime_type in config.images:
            # Wrap as a data URI — cheapest path that works with every
            # OpenAI-compatible server (including Ollama's /v1 proxy).
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type or 'image/jpeg'};base64,{b64_data}",
                    },
                }
            )
        parts.append({"type": "text", "text": prompt})
        return parts

    def _build_messages(self, prompt: str, config: LLMConfig) -> list[dict[str, Any]]:
        """Build OpenAI messages array from prompt and config."""
        messages: list[dict[str, Any]] = []
        if config.system_message and not _is_o_series(config.model):
            messages.append({"role": "system", "content": config.system_message})
        messages.append({"role": "user", "content": self._build_user_content(prompt, config)})
        return messages

    def _build_params(
        self,
        config: LLMConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build request parameters from config."""
        params: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
        }

        if not _is_o_series(config.model):
            params["temperature"] = config.temperature
        if config.max_output_tokens:
            params["max_tokens"] = config.max_output_tokens
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            params["presence_penalty"] = config.presence_penalty
        if tools:
            params["tools"] = tools
        if response_format:
            params["response_format"] = response_format
        if config.stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}

        # v0.4.0: thread timeouts through to the openai SDK. The SDK
        # accepts ``timeout=`` on every request; passing ``httpx.Timeout``
        # is honoured through its underlying httpx transport. Leave
        # unset when neither connect_timeout nor read_timeout is
        # configured so the SDK default keeps applying.
        timeout = self._resolve_httpx_timeout(config)
        if timeout is not None:
            params["timeout"] = timeout

        # v0.6.0: provider-specific body escape hatch. The openai Python
        # SDK splices ``extra_body`` into the outgoing JSON; vLLM /
        # OpenAI-compat servers use this for knobs like
        # ``chat_template_kwargs`` (Gemma-4 thinking toggle),
        # ``repetition_penalty``, ``top_k``, etc. — anything the SDK
        # doesn't model natively. Pass-through only; we don't validate
        # the keys because the set is server-specific.
        if config.extra_body:
            params["extra_body"] = dict(config.extra_body)

        # Auto-translate ``thinking_enabled`` into Gemma-4 / vLLM's
        # ``chat_template_kwargs.enable_thinking`` so users don't have to
        # hand-splice ``extra_body`` for the common case. Explicit caller
        # values in ``extra_body.chat_template_kwargs.enable_thinking``
        # win — we never overwrite.
        if config.thinking_enabled:
            current = params.get("extra_body") or {}
            ctk = current.get("chat_template_kwargs") or {}
            if "enable_thinking" not in ctk:
                ctk = {**ctk, "enable_thinking": True}
                current = {**current, "chat_template_kwargs": ctk}
                params["extra_body"] = current

        return params

    async def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            models = await client.models.list()
            return sorted([m.id for m in models.data])
        except Exception:
            return list(OPENAI_MODELS)

    def estimate_token_count(self, text: str, model: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)

    def prepare_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        client = self._get_client()
        messages = self._build_messages(prompt, config)
        params = self._build_params(config, messages)

        try:
            if config.stream:
                return self._stream_text(client, params)
            else:
                response = await client.chat.completions.create(**params)
                choice = response.choices[0]
                content = choice.message.content or ""
                if not content:
                    # Some OpenAI-compatible reasoning models leave content
                    # empty and place the answer in a ``reasoning`` field.
                    content = _extract_reasoning_text(choice.message)
                return TokenResponse(
                    content=content,
                    stop_reason=choice.finish_reason,
                )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def _stream_text(
        self, client: Any, params: dict[str, Any]
    ) -> AsyncIterator[TokenResponse]:
        # v0.7.0: poll the provider-level cancellation contextvar between
        # chunks. When the engine calls ``runner.stop(flow_id)`` (triggered
        # by an SSE client disconnect, a ``with qm.run.stream(...) as stream:
        # break``, or an explicit ``qm.Cancelled`` exception in a tool) it
        # flips the check; we then close the openai ``AsyncStream`` — which
        # closes the underlying httpx response — so vLLM / Ollama stops
        # generating and releases the slot. Without this the httpx
        # connection kept draining tokens until the model finished naturally.
        from quartermaster_providers.cancellation import should_cancel

        try:
            stream = await client.chat.completions.create(**params)
            try:
                async for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        finish_reason = chunk.choices[0].finish_reason
                        if delta and delta.content:
                            yield TokenResponse(
                                content=delta.content,
                                stop_reason=finish_reason,
                            )
                        else:
                            # OpenAI-compatible reasoning models stream their
                            # answer through ``reasoning_content`` (or ``reasoning``)
                            # when ``content`` is absent — surface that as visible
                            # text so callers don't see an empty stream.
                            reasoning_chunk = _extract_reasoning_text(delta) if delta else ""
                            if reasoning_chunk:
                                yield TokenResponse(
                                    content=reasoning_chunk,
                                    stop_reason=finish_reason,
                                )
                            elif finish_reason:
                                yield TokenResponse(
                                    content="",
                                    stop_reason=finish_reason,
                                )
                    if hasattr(chunk, "usage") and chunk.usage:
                        yield TokenResponse(
                            content="",
                            stop_reason="usage",
                        )
                    if should_cancel():
                        # Close the openai AsyncStream (→ httpx response)
                        # so the server knows to stop generating. The
                        # ``stop_reason="cancelled"`` sentinel tells the
                        # engine loop to treat this as a cooperative
                        # cancel rather than a natural finish.
                        await _aclose_stream(stream)
                        yield TokenResponse(content="", stop_reason="cancelled")
                        return
            finally:
                # Defensive: if the consumer abandons the generator mid-
                # stream (e.g. the engine raises out of the async-for),
                # still close the underlying httpx response rather than
                # leaking it into GC.
                await _aclose_stream(stream)
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def generate_tool_parameters(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        config: LLMConfig,
    ) -> ToolCallResponse:
        client = self._get_client()
        messages = self._build_messages(prompt, config)
        prepared_tools = [self.prepare_tool(t) for t in tools]
        params = self._build_params(config, messages, tools=prepared_tools)
        params.pop("stream", None)
        params.pop("stream_options", None)

        try:
            response = await client.chat.completions.create(**params)
            choice = response.choices[0]
            message = choice.message

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        parameters = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        parameters = {"raw": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            tool_id=tc.id,
                            parameters=parameters,
                        )
                    )

            usage = None
            if response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )

            text_content = message.content or ""
            # v0.6.0: text-form tool-call fallback. See
            # ``generate_native_response`` for the full rationale.
            if not tool_calls and text_content:
                salvaged, residual = _parse_text_form_tool_calls(text_content)
                if salvaged:
                    tool_calls = salvaged
                    text_content = residual

            return ToolCallResponse(
                text_content=text_content,
                tool_calls=tool_calls,
                stop_reason=choice.finish_reason,
                usage=usage,
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def generate_native_response(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        config: LLMConfig | None = None,
    ) -> NativeResponse:
        if config is None:
            raise InvalidRequestError("config is required", provider=self.PROVIDER_NAME)

        client = self._get_client()
        messages = self._build_messages(prompt, config)
        prepared_tools = [self.prepare_tool(t) for t in tools] if tools else None
        params = self._build_params(config, messages, tools=prepared_tools)
        params.pop("stream", None)
        params.pop("stream_options", None)

        try:
            response = await client.chat.completions.create(**params)
            choice = response.choices[0]
            message = choice.message

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        parameters = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        parameters = {"raw": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            tool_id=tc.id,
                            parameters=parameters,
                        )
                    )

            usage = None
            if response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )

            text_content = message.content or ""
            if not text_content:
                text_content = _extract_reasoning_text(message)

            # v0.6.0: text-form tool-call fallback. A vLLM / Ollama
            # server launched WITHOUT ``--tool-call-parser <flavour>``
            # leaks the chat-template's literal ``<|tool_call|>...``
            # sentinels into ``message.content`` rather than converting
            # them into structured ``tool_calls``. We used to miss those
            # entirely — the agent loop saw text, no tool_calls, and
            # exited thinking it had a final answer. Salvage them.
            if not tool_calls and text_content:
                salvaged, residual = _parse_text_form_tool_calls(text_content)
                if salvaged:
                    tool_calls = salvaged
                    text_content = residual

            return NativeResponse(
                text_content=text_content,
                thinking=[],
                tool_calls=tool_calls,
                stop_reason=choice.finish_reason,
                usage=usage,
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def generate_structured_response(
        self,
        prompt: str,
        response_schema: dict[str, Any] | type,
        config: LLMConfig,
    ) -> StructuredResponse:
        client = self._get_client()
        messages = self._build_messages(prompt, config)

        schema_dict: dict[str, Any]
        if isinstance(response_schema, type):
            if hasattr(response_schema, "__annotations__"):
                schema_dict = {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in response_schema.__annotations__},
                }
            else:
                schema_dict = {"type": "object"}
        else:
            schema_dict = response_schema

        json_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema: {json.dumps(schema_dict)}"
        )
        messages[-1]["content"] = json_prompt

        response_format = {"type": "json_object"}
        params = self._build_params(config, messages, response_format=response_format)
        params.pop("stream", None)
        params.pop("stream_options", None)

        try:
            response = await client.chat.completions.create(**params)
            choice = response.choices[0]
            raw_output = choice.message.content or ""

            try:
                structured = json.loads(raw_output)
            except json.JSONDecodeError:
                structured = {"raw": raw_output}

            usage = None
            if response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )

            return StructuredResponse(
                structured_output=structured,
                raw_output=raw_output,
                stop_reason=choice.finish_reason,
                usage=usage,
            )
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    async def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        client = self._get_client()
        try:
            with open(audio_path, "rb") as audio_file:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            return cast(str, response.text)
        except (AuthenticationError, RateLimitError, ProviderError):
            raise
        except Exception as e:
            self._handle_api_error(e)

    def get_cost_per_1k_input_tokens(self, model: str) -> float | None:
        pricing = OPENAI_PRICING.get(model)
        return pricing["input"] if pricing else None

    def get_cost_per_1k_output_tokens(self, model: str) -> float | None:
        pricing = OPENAI_PRICING.get(model)
        return pricing["output"] if pricing else None
