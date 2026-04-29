"""v0.8.2 — when an agent loop exits with no visible text but tools
were dispatched, force one more LLM call without tools so the model
must produce a final-answer text turn.

Symptom diagnosed in printer-app's Sora chat: Gemma-4 occasionally
finishes a turn by returning ``tool_calls=[]`` AND empty
``text_content`` on the iteration AFTER it dispatched tools — the
model has decided "I'm done" but didn't write the answer. Pre-v0.8.2
the agent's ``final_text`` stayed empty and ``FlowResult.final_output``
fell back to the latest non-empty finished node, which on a
``User → Agent → End`` graph is the User node — so ``result.text``
echoed the user's own question back as if it were the answer.

Operator-facing repro from production logs (2026-04-29):
    user:      "katera naročila imamo odprta"
    assistant: "katera naročila imamo odprta"   ← echo, not an answer

Fix: detect ``final_text == "" and tool_call_log != []`` at end of
loop, re-run the LAST iteration's prompt (which already carries the
``<tool_execution_log>`` block ending with "Use the tool results
above to produce your final answer.") with ``tools=None`` so the
model can only emit text. Adds at most one extra LLM call per agent
run and only when the symptom is present.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from quartermaster_providers import ProviderRegistry
from quartermaster_providers.base import AbstractLLMProvider
from quartermaster_providers.types import NativeResponse, ToolCall

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.example_runner import AgentExecutor
from quartermaster_engine.types import GraphNode, GraphSpec, NodeType


class _ScriptedProvider(AbstractLLMProvider):
    """Provider that returns a scripted sequence of NativeResponses,
    one per agent-loop iteration. Records every call so tests can
    assert about the forced-final-answer branch."""

    PROVIDER_NAME = "scripted"

    def __init__(self, responses: list[NativeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def list_models(self) -> list[str]:
        return ["scripted"]

    def estimate_token_count(self, text: str, model: str) -> int:
        return len(text)

    def prepare_tool(self, tool: Any) -> Any:
        return tool

    async def generate_text_response(self, prompt, config, history=None):
        raise NotImplementedError

    async def generate_tool_parameters(self, prompt, tools, config, history=None):
        raise NotImplementedError

    async def generate_native_response(self, prompt, tools=None, config=None, history=None):
        self.calls.append({"prompt": prompt, "tools": tools, "history": history})
        if not self._responses:
            return NativeResponse(text_content="", thinking=[], tool_calls=[], stop_reason="stop")
        return self._responses.pop(0)

    async def stream_native_response(
        self, prompt, tools=None, config=None, on_token=None, history=None
    ):
        # Delegate to the non-streaming path for tests; emit the buffered
        # text once at the end via on_token to mimic the base shim.
        resp = await self.generate_native_response(prompt, tools, config, history=history)
        if on_token is not None and resp.text_content:
            on_token(resp.text_content)
        return resp

    async def generate_structured_response(self, prompt, response_schema, config):
        raise NotImplementedError

    async def transcribe(self, audio_path):
        raise NotImplementedError


def _make_tool_registry():
    """Real ``ToolRegistry`` from quartermaster-tools registering one
    ``list_orders`` callable so the agent's ``_tool_definitions`` finds
    it via ``to_openai_tools()``."""
    from quartermaster_tools import ToolRegistry, tool

    @tool()
    def list_orders() -> dict:
        """List open work orders."""
        return {"count": 0, "orders": []}

    reg = ToolRegistry()
    reg.register(list_orders)
    return reg


def _make_agent_ctx() -> ExecutionContext:
    node = GraphNode(
        id=uuid4(),
        type=NodeType.AGENT,
        name="Assistant",
        metadata={
            "llm_provider": "scripted",
            "llm_model": "scripted",
            "program_version_ids": ["list_orders"],
        },
    )
    graph = GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=node.id,
        nodes=[node],
        edges=[],
    )
    return ExecutionContext(
        flow_id=uuid4(),
        node_id=node.id,
        graph=graph,
        current_node=node,
        messages=[],
        memory={"__user_input__": "katera naročila imamo odprta"},
        metadata={},
    )


def _registry_with(provider: AbstractLLMProvider) -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register_instance("scripted", provider)
    return reg


# ── Tests ────────────────────────────────────────────────────────────


class TestForcedFinalAnswer:
    def test_empty_text_after_tool_dispatch_triggers_extra_call(self) -> None:
        """The bug repro: iter 1 fires tool_calls, iter 2 returns
        empty text + empty tool_calls. v0.8.2 must force iter 3 with
        tools=None and surface its text answer."""
        provider = _ScriptedProvider(
            responses=[
                # Iter 1: model dispatches one tool call, no text.
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[ToolCall(tool_name="list_orders", tool_id="t0", parameters={})],
                    stop_reason="tool_calls",
                ),
                # Iter 2: model returns nothing — empty text, no tool_calls.
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                ),
                # Forced final-answer call: model now produces text.
                NativeResponse(
                    text_content="Imamo 0 odprtih naročil.",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                ),
            ]
        )
        executor = AgentExecutor(_registry_with(provider), tool_registry=_make_tool_registry())
        ctx = _make_agent_ctx()

        result = asyncio.run(executor.execute(ctx))

        assert result.success, result.error
        assert result.output_text == "Imamo 0 odprtih naročil."
        # Three LLM calls total: iter 1, iter 2, forced final.
        assert len(provider.calls) == 3
        # The forced call passed tools=None (no further tool dispatch allowed).
        assert provider.calls[-1]["tools"] is None

    def test_no_force_when_text_was_produced(self) -> None:
        """Happy path: the model returned text on the final iteration.
        The forced-call branch must NOT fire."""
        provider = _ScriptedProvider(
            responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[ToolCall(tool_name="list_orders", tool_id="t0", parameters={})],
                    stop_reason="tool_calls",
                ),
                NativeResponse(
                    text_content="Imamo 5 odprtih naročil.",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                ),
            ]
        )
        executor = AgentExecutor(_registry_with(provider), tool_registry=_make_tool_registry())
        ctx = _make_agent_ctx()

        result = asyncio.run(executor.execute(ctx))

        assert result.output_text == "Imamo 5 odprtih naročil."
        # Exactly two LLM calls — no forced extra.
        assert len(provider.calls) == 2

    def test_no_force_when_no_tools_were_dispatched(self) -> None:
        """If the agent finished empty WITHOUT calling any tools, the
        forced-call branch shouldn't fire — it'd just produce another
        empty response. Empty + no-tools is a different bug class
        (model genuinely had nothing to say) and should fall through
        to whatever salvage the consumer has."""
        provider = _ScriptedProvider(
            responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                ),
            ]
        )
        executor = AgentExecutor(_registry_with(provider), tool_registry=_make_tool_registry())
        ctx = _make_agent_ctx()

        result = asyncio.run(executor.execute(ctx))

        # Empty output, no force, exactly one call.
        assert result.output_text == ""
        assert len(provider.calls) == 1

    def test_forced_call_failure_falls_through_gracefully(self) -> None:
        """If the forced final-answer call raises, the agent should
        still return ``success=True`` with whatever final_text it had
        (empty here) — best-effort, not load-bearing."""

        class _RaisingProvider(_ScriptedProvider):
            async def stream_native_response(
                self, prompt, tools=None, config=None, on_token=None, history=None
            ):
                # First call works (returns the iter-1 tool dispatch).
                # Second call works (returns iter-2 empty).
                # Third call (the forced one with tools=None) raises.
                if tools is None:
                    raise RuntimeError("simulated provider blip")
                return await super().stream_native_response(
                    prompt, tools=tools, config=config, on_token=on_token, history=history
                )

        provider = _RaisingProvider(
            responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[ToolCall(tool_name="list_orders", tool_id="t0", parameters={})],
                    stop_reason="tool_calls",
                ),
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                ),
            ]
        )
        executor = AgentExecutor(_registry_with(provider), tool_registry=_make_tool_registry())
        ctx = _make_agent_ctx()

        result = asyncio.run(executor.execute(ctx))

        # Empty output, success=True, no exception bubbled up.
        assert result.success
        assert result.output_text == ""
