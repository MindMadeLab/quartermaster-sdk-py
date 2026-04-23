"""The v0.2.0 ergonomic graph runner.

Replaces the "import FlowRunner, build a node registry, construct the
runner, call .run()" dance with a one-liner:

    from quartermaster_sdk import Graph, run

    graph = Graph("chat").user().agent().build()
    result = run(graph, "Hello!")                 # sync
    for chunk in run.stream(graph, "Hello!"):    # streaming
        if chunk.type == "token":
            print(chunk.content, end="")

Provider is picked up from :func:`configure` or the
``provider_registry=`` kwarg; no need for the integrator to touch
``quartermaster_engine.FlowRunner`` or
``quartermaster_engine.build_default_registry`` directly.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

from quartermaster_engine import (
    CustomEvent,
    FlowError,
    FlowEvent,
    FlowFinished,
    FlowRunner,
    ImageInput,
    NodeFinished,
    NodeStarted,
    ProgressEvent,
    TokenGenerated,
    ToolCallFinished,
    ToolCallStarted,
    UserInputRequired,
    prepare_images,
)

from . import _listeners
from ._chunks import (
    AwaitInputChunk,
    Chunk,
    CustomChunk,
    DoneChunk,
    ErrorChunk,
    NodeFinishChunk,
    NodeStartChunk,
    ProgressChunk,
    TokenChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from ._config import get_auto_redact_config, get_default_registry, get_default_timeouts
from ._result import Result
from ._session import ChatTurn, SessionStore
from ._stream_filters import _Stream
from ._trace import Trace

if TYPE_CHECKING:
    from quartermaster_graph import GraphBuilder, GraphSpec
    from quartermaster_providers import ProviderRegistry

logger = logging.getLogger(__name__)


class StreamDeadlineExceeded(TimeoutError):
    """The stream's total wall-clock budget was exhausted.

    Raised by ``qm.run.stream(..., deadline_seconds=N)`` /
    ``qm.arun.stream(..., deadline_seconds=N)`` when the consumer
    loop hasn't terminated after *N* seconds. Subclasses
    :class:`TimeoutError` so callers that already ``except
    TimeoutError`` keep catching it. Added in v0.4.0.
    """


def _resolve_call_timeouts(
    *,
    timeout: float | None,
    connect_timeout: float | None,
    read_timeout: float | None,
) -> dict[str, float | None]:
    """Merge configure-time defaults with per-call overrides.

    Added in v0.4.0. Per-call overrides take precedence over the
    configure-time defaults; passing ``timeout=`` as a shortcut sets
    both phases for this call only.

    Raises ``ValueError`` when the caller passes both the shortcut
    and a specific field — same rule as :func:`configure`.
    """
    if timeout is not None and (
        connect_timeout is not None or read_timeout is not None
    ):
        raise ValueError(
            "run(): pass timeout= OR connect_timeout/read_timeout, not both."
        )
    for label, value in (
        ("timeout", timeout),
        ("connect_timeout", connect_timeout),
        ("read_timeout", read_timeout),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"run(): {label}= must be > 0, got {value!r}")

    defaults = get_default_timeouts()
    if timeout is not None:
        resolved_connect: float | None = float(timeout)
        resolved_read: float | None = float(timeout)
    else:
        resolved_connect = (
            float(connect_timeout)
            if connect_timeout is not None
            else defaults["connect_timeout"]
        )
        resolved_read = (
            float(read_timeout)
            if read_timeout is not None
            else defaults["read_timeout"]
        )
    return {"connect_timeout": resolved_connect, "read_timeout": resolved_read}


def _resolve_graph(graph: GraphBuilder | GraphSpec) -> GraphSpec:
    """Accept either a builder (auto-finalise) or a pre-built spec."""
    if hasattr(graph, "build"):
        return graph.build()
    return graph


def _extract_inline_tools(graph: GraphBuilder | GraphSpec) -> dict[str, Any]:
    """Return the builder-side ``_inline_tools`` dict if present.

    v0.4.0: ``.agent(tools=[...])`` stashes @tool()-decorated callables
    (and auto-decorated bare functions) on the builder so the runner can
    merge them into the run-scoped tool registry without requiring the
    caller to do a manual ``get_default_registry().register(...)`` dance.

    Returns an empty dict when the graph is a pre-built :class:`GraphSpec`
    (callables are not JSON/YAML-serialisable so they can't survive a
    build-then-transport round-trip — callers in that mode must register
    their tools on the ``tool_registry=`` they pass in, as before).
    """
    raw = getattr(graph, "_inline_tools", None)
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _extract_retry_predicates(graph: GraphBuilder | GraphSpec) -> dict[str, Any]:
    """Return the builder-side ``_retry_predicates`` dict if present.

    v0.7.0: ``.agent(retry={"on": <callable>})`` (and the analogous
    ``instruction()`` / ``instruction_form()`` forms) stashes the
    predicate on the builder's ``_retry_predicates`` side-channel so the
    engine can resolve it by node name at run time. Callables cannot
    survive JSON/YAML so they do NOT round-trip through a rebuilt
    :class:`GraphSpec` — callers who transport graphs across process
    boundaries must re-attach their predicates via a freshly-invoked
    builder on the consumer side.

    Returns an empty dict when the graph carries no predicates (mirrors
    :func:`_extract_inline_tools`).
    """
    raw = getattr(graph, "_retry_predicates", None)
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _merge_inline_tools(
    tool_registry: Any | None,
    inline_tools: dict[str, Any],
) -> Any | None:
    """Return a tool registry that exposes *inline_tools* plus any
    registry the caller passed in.

    Strategy:

    * If there are no inline callables, return *tool_registry* unchanged.
    * If the caller supplied a registry AND we have inline tools, wrap
      both in a combining shim that first consults the inline dict and
      then delegates to the caller registry. We don't mutate the caller
      registry — the global default stays pristine so tests can't leak
      tool names across runs.
    * If the caller supplied nothing AND we have inline tools, build a
      fresh :class:`ToolRegistry` (or fall back to the shim if
      quartermaster-tools is unavailable, though that branch is
      defensive only — quartermaster-tools is a hard dep).
    """
    if not inline_tools:
        return tool_registry

    if tool_registry is None:
        try:
            from quartermaster_tools import ToolRegistry
        except ImportError:  # pragma: no cover — quartermaster-tools is a hard dep
            return _InlineToolRegistry(inline_tools, None)
        reg = ToolRegistry()
        for tool in inline_tools.values():
            try:
                reg.register(tool)
            except ValueError:
                # Same name/version already in this fresh registry — skip
                # the duplicate. Can happen if the same tool is referenced
                # across multiple agent nodes in the same graph.
                pass
        return reg

    return _InlineToolRegistry(inline_tools, tool_registry)


class _InlineToolRegistry:
    """Per-run shim that exposes inline @tool() callables AND delegates
    unknown lookups to the caller-provided registry.

    Mirrors the subset of :class:`quartermaster_tools.ToolRegistry`
    interface that the engine's ``AgentExecutor`` actually calls:

    * ``get(name) → tool`` — raises ``KeyError`` if neither the inline
      dict nor the delegate knows the name.
    * ``to_openai_tools() / to_anthropic_tools() / to_mcp_tools() /
      list_tools()`` — concatenates descriptors from both sides so the
      agent's JSON schema catalog includes both inline and pre-registered
      tools.

    The original caller-supplied registry is left untouched; inline
    entries live in this shim for the lifetime of the run and are
    garbage-collected when the shim goes out of scope.
    """

    def __init__(self, inline: dict[str, Any], delegate: Any | None) -> None:
        self._inline = dict(inline)
        self._delegate = delegate

    def get(self, name: str, version: str | None = None) -> Any:
        if name in self._inline:
            return self._inline[name]
        if self._delegate is None:
            raise KeyError(name)
        if version is None:
            return self._delegate.get(name)
        return self._delegate.get(name, version)

    def __contains__(self, name: str) -> bool:
        if name in self._inline:
            return True
        if self._delegate is None:
            return False
        return name in self._delegate

    def list_tools(self) -> list[Any]:
        out: list[Any] = [tool.info() for tool in self._inline.values()]
        if self._delegate is not None:
            delegate_list = getattr(self._delegate, "list_tools", None)
            if callable(delegate_list):
                out.extend(delegate_list())
        return out

    def _collect_schemas(self, method: str) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        # Inline side: reuse ToolRegistry's schema helpers by temporarily
        # building a throwaway ToolRegistry. Keeps the schema format in
        # sync with what the canonical registry emits.
        if self._inline:
            try:
                from quartermaster_tools import ToolRegistry

                tmp = ToolRegistry()
                for tool in self._inline.values():
                    try:
                        tmp.register(tool)
                    except ValueError:
                        pass
                fn = getattr(tmp, method, None)
                if callable(fn):
                    schemas.extend(fn())
            except ImportError:  # pragma: no cover
                pass
        if self._delegate is not None:
            fn = getattr(self._delegate, method, None)
            if callable(fn):
                schemas.extend(fn())
        return schemas

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return self._collect_schemas("to_openai_tools")

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return self._collect_schemas("to_anthropic_tools")

    def to_mcp_tools(self) -> list[dict[str, Any]]:
        return self._collect_schemas("to_mcp_tools")


def _apply_pii_redaction(text: str, policy: str) -> str:
    """Apply PII redaction to *text* using the built-in privacy tools.

    Called when ``qm.configure(auto_redact_pii=True)`` is set.  The
    ``policy`` string controls which entity types are redacted — ``"all"``
    (default) strips every detected PII class; a comma-separated list
    (``"email,phone,credit_card"``) restricts to those types.

    If ``quartermaster-tools`` isn't installed or the privacy tools fail,
    falls back to the original text with a warning — never crashes the
    flow just because redaction couldn't run.
    """
    try:
        from quartermaster_tools.builtin.privacy.redact import RedactPIITool
    except ImportError:
        logger.warning(
            "auto_redact_pii enabled but quartermaster-tools is not installed; "
            "skipping redaction"
        )
        return text
    try:
        result = RedactPIITool.run(text=text, strategy="redact")
        return result.data.get("redacted_text", text) if result.success else text
    except Exception:
        logger.warning("PII redaction failed; passing original text", exc_info=True)
        return text


def assert_traces_equal(
    actual: Trace,
    recorded: Trace,
    *,
    ignore: list[str] | None = None,
) -> None:
    """Assert that *actual* and *recorded* traces match on tool calls.

    Compares the sequence of tool calls (tool name + arguments) between
    two traces.  Fields listed in *ignore* are excluded from comparison
    — common values: ``"timestamps"``, ``"node_ids"``, ``"results"``.

    Raises ``AssertionError`` with a diff when traces diverge.  Designed
    for regression tests that capture a known-good trace via
    ``result.trace.as_jsonl()`` and replay it later to detect silent
    tool-call drift.

    Added in v0.4.0.
    """
    ignore_set = set(ignore or [])

    def _extract_calls(trace: Trace) -> list[dict]:
        calls = []
        for tc in trace.tool_calls:
            entry = {"tool": tc.get("tool"), "arguments": tc.get("arguments")}
            if "results" not in ignore_set:
                entry["result"] = tc.get("result")
            calls.append(entry)
        return calls

    actual_calls = _extract_calls(actual)
    recorded_calls = _extract_calls(recorded)

    if actual_calls != recorded_calls:
        import json

        raise AssertionError(
            f"Trace tool-call drift detected.\n"
            f"  Recorded ({len(recorded_calls)} calls): "
            f"{json.dumps(recorded_calls, indent=2, default=str)}\n"
            f"  Actual   ({len(actual_calls)} calls): "
            f"{json.dumps(actual_calls, indent=2, default=str)}"
        )


class _RunCallable:
    """Callable + ``.stream`` method exposed as ``qm.run``.

    Packaged as a class so ``run`` and ``run.stream`` share the same
    argument-resolution path without repeating boilerplate.
    """

    def __call__(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        image: ImageInput | None = None,
        images: list[ImageInput] | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        session: SessionStore | None = None,
        session_id: str | None = None,
    ) -> Result:
        """Execute *graph* against *user_input* and return a :class:`Result`.

        Args:
            graph: Either a :class:`GraphBuilder` (auto-finalised) or
                a pre-built :class:`GraphSpec`.
            user_input: Primary user message injected into the graph.
            image: Optional single image input for vision-capable graphs.
                Accepts raw ``bytes``, a :class:`pathlib.Path`, or a
                filesystem path string. When set, the graph's
                ``.vision()`` node receives the image alongside
                *user_input*. Mutually exclusive with *images*. On
                graphs that don't declare a vision node this is a
                no-op — the image is silently ignored so callers don't
                need branch on "is this graph vision-enabled?".
            images: Optional list of image inputs (same per-item types
                as *image*). Use this when the graph's vision node
                should see multiple images in a single turn. Mutually
                exclusive with *image*.
            provider_registry: Override the configured default registry.
            tool_registry: Optional :class:`quartermaster_tools.ToolRegistry`
                made available to ``agent()``-type nodes that specify
                ``tools=[...]``.
            timeout: Per-call shortcut — same budget for both the
                connect and read phases of every LLM call in this
                flow. Overrides the ``qm.configure(timeout=...)``
                default. Added in v0.4.0. Mutually exclusive with
                ``connect_timeout`` / ``read_timeout``.
            connect_timeout: Per-call override for the connect phase.
                Added in v0.4.0.
            read_timeout: Per-call override for the read phase (per-
                LLM-call ceiling). Added in v0.4.0.
            session: Optional :class:`SessionStore` for multi-turn chat.
                When provided alongside *session_id*, the runner loads
                prior turns, folds them into *user_input*, and appends
                the new user + assistant turns after the run. Added in
                v0.4.0.
            session_id: Session key for the session store.  Required
                when *session* is set; ignored otherwise.
        """
        # ── v0.4.0 session: load history ──────────────────────────────
        if session is not None:
            if session_id is None:
                raise ValueError("run(): session= requires session_id=")
            history: list[ChatTurn] = session.load(session_id)
            if history:
                parts = [f"{t.role.capitalize()}: {t.content}" for t in history]
                parts.append(f"User: {user_input}")
                user_input = "\n".join(parts)

        # ── v0.4.0 auto-redact PII ───────────────────────────────────
        auto_redact, redact_policy = get_auto_redact_config()
        if auto_redact:
            user_input = _apply_pii_redaction(user_input, redact_policy)

        inline_tools = _extract_inline_tools(graph)
        retry_predicates = _extract_retry_predicates(graph)
        spec = _resolve_graph(graph)
        registry = provider_registry or get_default_registry()
        prepared_images = prepare_images(image=image, images=images)
        effective_tool_registry = _merge_inline_tools(tool_registry, inline_tools)
        llm_timeouts = _resolve_call_timeouts(
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

        # v0.3.0 trace: accumulate every FlowEvent into a local list
        # so we can build a ``Trace`` and attach it to the returned
        # ``Result``. Fires alongside the global listener dispatch so
        # bolt-on instrumentation still sees the same stream.
        trace_events: list[FlowEvent] = []

        def _collect(event: FlowEvent) -> None:
            trace_events.append(event)
            _listeners.dispatch(event)

        runner = FlowRunner(
            graph=spec,
            provider_registry=registry,
            tool_registry=effective_tool_registry,
            retry_predicates=retry_predicates or None,
            # Forward every event to the global listener registry so
            # bolt-on instrumentation (e.g. ``qm.telemetry.instrument()``)
            # observes the same FlowEvent stream the streaming runner
            # gets — even though there's no consumer queue here.
            on_event=_collect,
        )
        started = time.perf_counter()
        fr = runner.run(
            user_input,
            images=prepared_images or None,
            llm_timeouts=llm_timeouts,
        )
        elapsed = time.perf_counter() - started

        result = Result.from_flow_result(fr)
        # FlowResult's ``duration_seconds`` is authoritative when
        # populated; fall back to our wall-clock measurement for the
        # trace if the engine left it at 0.
        result.trace = Trace.from_events(
            trace_events,
            duration_seconds=fr.duration_seconds or elapsed,
        )
        result.trace.user_input = user_input

        # ── v0.4.0 session: persist new turns ────────────────────────
        if session is not None and session_id is not None:
            session.append(session_id, ChatTurn(role="user", content=user_input))
            session.append(session_id, ChatTurn(role="assistant", content=result.text))

        return result

    def stream(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        image: ImageInput | None = None,
        images: list[ImageInput] | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        deadline_seconds: float | None = None,
    ) -> _Stream:
        """Run the graph and yield typed :class:`Chunk` events as they arrive.

        Returns a :class:`_Stream` wrapper that is itself iterable (so
        existing ``for chunk in qm.run.stream(...)`` loops work
        unchanged) and exposes filter helpers — ``.tokens()``,
        ``.tool_calls()``, ``.progress()``, ``.custom(name=...)`` —
        for the common "pluck one chunk type out of the stream"
        patterns.

        Accepts the same *image* / *images* kwargs as :meth:`__call__` —
        pass a single image (``bytes`` / :class:`pathlib.Path` / path
        string) via *image*, or a list of images via *images*. They're
        forwarded to the vision node's ``LLMExecutor`` via flow memory.

        Terminates with a :class:`DoneChunk` (on success) or
        :class:`ErrorChunk` (on unrecoverable failure).  The graph
        executes on a background thread so the caller can iterate the
        yielded chunks synchronously — no ``async``/``await`` required.

        **Cancellation (v0.4.0):** the returned wrapper supports the
        context-manager protocol — ``with qm.run.stream(...) as s:``.
        On every exit path (normal completion, ``break``, ``return``,
        or exception) the wrapper calls :meth:`FlowRunner.stop` on the
        in-flight ``flow_id``, which the engine checks in
        ``_execute_node`` before dispatching any further work. Nodes
        already mid-flight finish their current LLM/tool call — no
        hard kill — but no new nodes are dispatched, so long agent
        loops unwind within a bounded time instead of leaking API
        costs in the background. Tools polling
        ``qm.current_context().cancelled`` observe ``True`` immediately
        after the ``with`` exit fires and can bail cooperatively.

        Legacy raw-iteration call sites keep their old behaviour — the
        generator's ``finally`` block still calls ``runner.stop`` when
        the iterator is abandoned (break from a plain ``for`` loop).

        Args (v0.4.0):
            timeout / connect_timeout / read_timeout: Per-LLM-call
                timeout overrides (see :meth:`__call__`).
            deadline_seconds: Total wall-clock budget for the entire
                stream. Raises :class:`StreamDeadlineExceeded` when
                exceeded. Independent of *read_timeout*.
        """
        # v0.4.0 stream cancellation: a mutable "stop handle" dict the
        # generator populates with (runner, flow_id) once it boots.
        # The context-manager exit callback reads from this cell so it
        # can call :meth:`FlowRunner.stop` even if the caller opened
        # the ``with`` block but hasn't iterated yet. Closing over the
        # generator's local ``runner`` directly wouldn't work because
        # the generator body doesn't run until the first ``next()``.
        stop_handle: dict[str, Any] = {}

        def _on_exit() -> None:
            runner = stop_handle.get("runner")
            fid = stop_handle.get("flow_id")
            if runner is None or fid is None:
                # Generator never started — nothing to stop. Legitimate
                # when the caller opens the ``with`` but exits before
                # the first iteration.
                return
            try:
                runner.stop(fid)
            except Exception:  # pragma: no cover — defensive
                logger.exception(
                    "run.stream: runner.stop(%s) raised from context-manager exit",
                    fid,
                )

        return _Stream(
            self._iter_chunks(
                graph=graph,
                user_input=user_input,
                image=image,
                images=images,
                provider_registry=provider_registry,
                tool_registry=tool_registry,
                timeout=timeout,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                deadline_seconds=deadline_seconds,
                stop_handle=stop_handle,
            ),
            on_exit=_on_exit,
        )

    def _iter_chunks(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        image: ImageInput | None = None,
        images: list[ImageInput] | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        deadline_seconds: float | None = None,
        stop_handle: dict[str, Any] | None = None,
    ) -> Iterator[Chunk]:
        """Inner generator that powers :meth:`stream`.

        Kept private: callers always go through ``stream()`` which
        wraps the result in :class:`_Stream` for filter support.

        *stop_handle* is the mutable cell the context-manager exit
        callback in :meth:`stream` reads to fire ``runner.stop``. We
        populate it with the live runner + flow_id as soon as both
        exist so the callback has something to act on even if the
        caller breaks out of the ``with`` before any events arrive.
        """
        if deadline_seconds is not None and deadline_seconds <= 0:
            raise ValueError(
                f"run.stream(): deadline_seconds must be > 0, got {deadline_seconds!r}"
            )
        inline_tools = _extract_inline_tools(graph)
        retry_predicates = _extract_retry_predicates(graph)
        spec = _resolve_graph(graph)
        registry = provider_registry or get_default_registry()
        prepared_images = prepare_images(image=image, images=images)
        effective_tool_registry = _merge_inline_tools(tool_registry, inline_tools)
        llm_timeouts = _resolve_call_timeouts(
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

        q: queue.Queue[FlowEvent | None] = queue.Queue()
        holder_lock = threading.Lock()
        holder: dict[str, Any] = {}
        # Pre-generate the flow_id so cancellation can call
        # ``runner.stop(flow_id)`` even if the caller breaks out of the
        # iterator before the first event fires.  ``FlowRunner.run``
        # accepts an optional ``flow_id=`` and will use ours instead of
        # minting its own — this is the documented public API, not
        # reaching into a private attribute.
        flow_id = uuid4()

        # v0.3.0 trace: same collector the sync path uses, populated in
        # arrival order on the runner thread. Safe to append from the
        # event thread and read from the iterator thread because the
        # final ``Trace`` is only built AFTER the runner thread joins.
        trace_events: list[FlowEvent] = []

        def on_event(event: FlowEvent) -> None:
            # Fan out to bolt-on instrumentation (telemetry, custom
            # listeners, etc.) BEFORE queuing for the iterator — keeps
            # the listener wall-clock close to the original event time.
            _listeners.dispatch(event)
            trace_events.append(event)
            q.put(event)

        runner = FlowRunner(
            graph=spec,
            provider_registry=registry,
            tool_registry=effective_tool_registry,
            retry_predicates=retry_predicates or None,
            on_event=on_event,
        )

        # v0.4.0: publish the live runner+flow_id to the
        # context-manager exit callback's stop_handle the moment both
        # exist. Any break/return/exception during the ``with`` block
        # now routes through ``_on_exit`` into ``runner.stop(flow_id)``.
        if stop_handle is not None:
            stop_handle["runner"] = runner
            stop_handle["flow_id"] = flow_id

        def _run_thread() -> None:
            try:
                fr = runner.run(
                    user_input,
                    images=prepared_images or None,
                    flow_id=flow_id,
                    llm_timeouts=llm_timeouts,
                )
                with holder_lock:
                    holder["result"] = fr
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("run.stream: runner.run raised")
                with holder_lock:
                    holder["exception"] = exc
            finally:
                # Sentinel: signal the iterator loop that there are no
                # more events.  Placed in ``finally`` so a runner crash
                # doesn't hang the caller.
                q.put(None)

        thread = threading.Thread(target=_run_thread, name="qm-run-stream", daemon=True)
        started = time.perf_counter()
        thread.start()

        # v0.4.0: total wall-clock ceiling for the whole stream.
        # Independent of ``read_timeout`` (per-LLM-call). Computed
        # once so a stalled ``q.get`` doesn't silently push the
        # budget out.
        deadline_at: float | None = (
            time.monotonic() + deadline_seconds
            if deadline_seconds is not None
            else None
        )

        try:
            while True:
                # Short timeout so the caller can still get control back
                # (e.g. to abort, log progress) if the runner stalls.
                if deadline_at is not None:
                    remaining = deadline_at - time.monotonic()
                    if remaining <= 0:
                        raise StreamDeadlineExceeded(
                            f"run.stream: exceeded deadline_seconds={deadline_seconds}"
                        )
                    try:
                        event = q.get(timeout=remaining)
                    except queue.Empty:
                        raise StreamDeadlineExceeded(
                            f"run.stream: exceeded deadline_seconds={deadline_seconds}"
                        ) from None
                else:
                    try:
                        event = q.get(timeout=300.0)
                    except queue.Empty:
                        logger.warning(
                            "run.stream: no event for 5 minutes; still waiting"
                        )
                        continue
                if event is None:
                    break
                chunk = _event_to_chunk(event)
                if chunk is not None:
                    yield chunk
        finally:
            # Caller abandoned the iterator — tell the runner to stop
            # the flow by its pre-generated id.  ``_execute_node``
            # short-circuits on the next dispatch, so nodes mid-LLM-call
            # finish (no hard kill) but no new work is scheduled.
            if thread.is_alive():
                try:
                    runner.stop(flow_id)
                except Exception:  # pragma: no cover — defensive
                    logger.exception("run.stream: runner.stop(%s) raised", flow_id)
            thread.join(timeout=5.0)

        with holder_lock:
            exception = holder.get("exception")
            fr = holder.get("result")

        if exception is not None:
            yield ErrorChunk(error=str(exception))
            return

        if fr is None:
            yield ErrorChunk(error="run.stream: runner produced no result")
            return

        elapsed = time.perf_counter() - started
        result = Result.from_flow_result(fr)
        # v0.3.0: populate the structured trace from the accumulated
        # event stream. The collector callback appended every FlowEvent
        # as it fired; we build the bucketed/by-node view once and
        # attach before handing the caller the DoneChunk.
        result.trace = Trace.from_events(
            trace_events,
            duration_seconds=fr.duration_seconds or elapsed,
        )
        yield DoneChunk(result=result)


def _event_to_chunk(event: FlowEvent) -> Chunk | None:
    """Translate an engine :class:`FlowEvent` into a public :class:`Chunk`.

    Returns ``None`` for events we intentionally swallow (they fold into
    ``DoneChunk`` at the end of the stream).
    """
    if isinstance(event, TokenGenerated):
        return TokenChunk(content=event.token)
    if isinstance(event, NodeStarted):
        return NodeStartChunk(
            node_name=event.node_name,
            node_type=event.node_type.value,
        )
    if isinstance(event, NodeFinished):
        return NodeFinishChunk(
            node_name=getattr(event, "node_name", ""),
            output=event.result or "",
        )
    if isinstance(event, ToolCallStarted):
        # Live "tool is being called now" card — fires before the tool
        # actually runs.  Arguments are passed straight through so UIs
        # can render them the moment the call is dispatched, instead of
        # waiting for the final NodeFinish / Done event.
        return ToolCallChunk(tool=event.tool, args=dict(event.arguments))
    if isinstance(event, ToolCallFinished):
        # Paired with ToolCallStarted.  ``result`` is the string the LLM
        # sees next turn (``"[ERROR: ...]"`` sentinel on failure);
        # ``raw`` is the structured payload when the tool returned one,
        # ``None`` otherwise.  ``error`` is non-None only on failure —
        # UIs can use it as the "red card" trigger.
        return ToolResultChunk(
            tool=event.tool,
            result=event.raw if event.raw is not None else event.result,
            error=event.error,
        )
    if isinstance(event, ProgressEvent):
        # Application-emitted progress signal (e.g. from inside a
        # long-running tool). Interleaves with TokenChunk on the
        # consumer's iterator — UIs can render "searching…" /
        # "parsing…" cards without interrupting the model tokens.
        return ProgressChunk(
            message=event.message,
            percent=event.percent,
            data=dict(event.data),
        )
    if isinstance(event, CustomEvent):
        # Caller-tagged structured event — consumers filter via
        # ``stream.custom(name="retrieved_docs")`` for the specific
        # milestone they care about.
        return CustomChunk(
            name=event.name,
            payload=dict(event.payload),
        )
    if isinstance(event, UserInputRequired):
        return AwaitInputChunk(
            prompt=event.prompt,
            options=list(event.options or []),
        )
    if isinstance(event, FlowError):
        return ErrorChunk(error=event.error)
    if isinstance(event, FlowFinished):
        # DoneChunk is emitted explicitly after the thread joins so the
        # caller gets the full Result (not just the final_output string
        # this event carries).
        return None
    return None


#: Publicly exported runner.  ``qm.run(graph, input)`` runs sync;
#: ``qm.run.stream(graph, input)`` yields typed Chunk objects.
run = _RunCallable()


__all__ = ["run", "Result", "Chunk"]
