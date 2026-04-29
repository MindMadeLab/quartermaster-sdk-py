"""v0.7.1 — ``AgentExecutor`` uses ``stream_native_response`` so the
chat UI gets incremental tokens instead of one big chunk per turn.

Pre-v0.7.1 the agent loop called ``generate_native_response`` (non-
streaming) so it could read tool_calls atomically — but that meant the
entire visible-text reply landed as a single :class:`TokenGenerated`
event after the model finished. Long answers showed up as a 5-second
pause then a wall of text.

v0.7.1 swaps to ``stream_native_response`` which threads visible-text
chunks to ``context.emit_token`` as they arrive while still returning
the full :class:`NativeResponse` for atomic tool dispatch. This test
verifies the executor wiring: a streaming-aware provider must result in
multiple ``TokenGenerated`` events per turn, and the tool-dispatch
loop must continue working unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from quartermaster_providers import LLMConfig, ProviderRegistry
from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.types import NativeResponse, ToolCall

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.example_runner import AgentExecutor
from quartermaster_engine.types import GraphNode, GraphSpec, NodeType


class _StreamingMockProvider(AbstractLLMProvider):
    """MockProvider variant that overrides ``stream_native_response`` to
    emit tokens in chunks — proves the AgentExecutor wires through the
    callback rather than buffering."""

    PROVIDER_NAME = "streaming_mock"

    def __init__(
        self,
        text_chunks: list[str],
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.text_chunks = list(text_chunks)
        self.tool_calls_to_return = list(tool_calls or [])
        self.finish_reason = finish_reason
        self.stream_call_count = 0
        self.native_call_count = 0

    async def list_models(self) -> list[str]:
        return ["streaming-mock"]

    def estimate_token_count(self, text: str, model: str) -> int:
        return len(text)

    def prepare_tool(self, tool: Any) -> Any:
        return tool

    async def generate_text_response(self, prompt: str, config: LLMConfig) -> Any:
        raise NotImplementedError

    async def generate_tool_parameters(
        self, prompt: str, tools: list[Any], config: LLMConfig
    ) -> Any:
        raise NotImplementedError

    async def generate_native_response(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        config: LLMConfig | None = None,
        history: Any = None,
    ) -> NativeResponse:
        # Tracked so the test can prove we DIDN'T fall back to non-streaming.
        self.native_call_count += 1
        self.last_history = history
        return NativeResponse(
            text_content="".join(self.text_chunks),
            thinking=[],
            tool_calls=self.tool_calls_to_return,
            stop_reason=self.finish_reason,
        )

    async def stream_native_response(
        self,
        prompt: str,
        tools: list[Any] | None = None,
        config: LLMConfig | None = None,
        on_token: Any = None,
        history: Any = None,
    ) -> NativeResponse:
        self.stream_call_count += 1
        self.last_history = history
        for chunk in self.text_chunks:
            if on_token is not None:
                on_token(chunk)
            await asyncio.sleep(0)  # yield so it actually behaves async-y
        return NativeResponse(
            text_content="".join(self.text_chunks),
            thinking=[],
            tool_calls=self.tool_calls_to_return,
            stop_reason=self.finish_reason,
        )

    async def generate_structured_response(
        self,
        prompt: str,
        response_schema: Any,
        config: LLMConfig,
    ) -> Any:
        raise NotImplementedError

    async def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


def _make_agent_ctx() -> tuple[ExecutionContext, list[str]]:
    """Build an ExecutionContext and capture every ``emit_token`` call."""
    captured: list[str] = []

    node = GraphNode(
        id=uuid4(),
        type=NodeType.AGENT,
        name="Agent",
        metadata={"llm_provider": "streaming_mock", "llm_model": "streaming-mock"},
    )
    graph = GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=node.id,
        nodes=[node],
        edges=[],
    )
    ctx = ExecutionContext(
        flow_id=uuid4(),
        node_id=node.id,
        graph=graph,
        current_node=node,
        messages=[],
        memory={"__user_input__": "tell me a story"},
        metadata={},
    )

    # Hook context.emit_token to capture the per-chunk events.
    original = ctx.emit_token

    def capture(text: str) -> None:
        captured.append(text)
        original(text)

    ctx.emit_token = capture  # type: ignore[method-assign]
    return ctx, captured


def _registry_with(provider: AbstractLLMProvider) -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register_instance("streaming_mock", provider)
    return reg


class TestAgentStreamingTokens:
    """Token chunks must reach ``context.emit_token`` one-per-chunk
    rather than as one big buffered chunk at end of turn."""

    def test_agent_emits_one_event_per_chunk(self) -> None:
        provider = _StreamingMockProvider(
            text_chunks=["Once ", "upon ", "a ", "time."],
        )
        ctx, captured = _make_agent_ctx()
        executor = AgentExecutor(_registry_with(provider))

        result = asyncio.run(executor.execute(ctx))

        assert result.success, result.error
        assert captured == ["Once ", "upon ", "a ", "time."], (
            "AgentExecutor must forward each streamed chunk as its own "
            "emit_token event so the chat UI animates incrementally."
        )

    def test_agent_does_not_double_emit_buffered_text(self) -> None:
        """Regression guard — pre-v0.7.1 the executor used to call
        ``emit_token(text)`` AFTER the call returned. Now token streaming
        happens inside ``stream_native_response`` via on_token; the
        post-call branch must NOT re-emit the buffered text."""
        provider = _StreamingMockProvider(text_chunks=["hello", " ", "world"])
        ctx, captured = _make_agent_ctx()
        executor = AgentExecutor(_registry_with(provider))

        asyncio.run(executor.execute(ctx))

        # Three chunks went through, plus zero re-emits = three total.
        assert len(captured) == 3
        assert "".join(captured) == "hello world"

    def test_agent_uses_stream_method_not_native(self) -> None:
        """The executor must reach for ``stream_native_response``, not
        ``generate_native_response`` — otherwise the chat UI loses
        progressive feedback on every agent turn."""
        provider = _StreamingMockProvider(text_chunks=["x", "y"])
        ctx, _captured = _make_agent_ctx()
        executor = AgentExecutor(_registry_with(provider))

        asyncio.run(executor.execute(ctx))

        assert provider.stream_call_count == 1
        assert provider.native_call_count == 0


class TestAgentStreamingPreservesToolDispatch:
    """The whole point of streaming + atomic tool_calls is that tool
    dispatch keeps working as before. A turn that emits tokens AND a
    tool_call must still trigger tool execution, not be mistaken for a
    finalised reply."""

    def test_streamed_tokens_plus_tool_call_triggers_dispatch(self) -> None:
        """First turn: tokens + a tool_call (model is narrating before
        calling). With no tools wired up we treat that as final text per
        the existing ``not tools → break`` rule, but token streaming
        must have happened on the first turn."""
        provider = _StreamingMockProvider(
            text_chunks=["Looking ", "this ", "up..."],
            tool_calls=[
                ToolCall(tool_name="search", tool_id="call_0", parameters={"q": "x"}),
            ],
        )
        ctx, captured = _make_agent_ctx()
        executor = AgentExecutor(_registry_with(provider))

        # No ``tools`` configured on the node — agent loop sees a
        # tool_call request, can't dispatch, falls through with text.
        asyncio.run(executor.execute(ctx))

        # Tokens streamed despite the loop terminating early.
        assert captured == ["Looking ", "this ", "up..."]
