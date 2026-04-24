"""v0.7.0 — structured-output LLM nodes use forced ``tool_choice`` so
the model returns via a typed tool call, not free-form text.

Why this is better than the v0.6.x prompt-engineering approach:

- The openai / vLLM / OpenAI-compat stack enforces the JSON shape at
  the token level via the tool-call parser. No fence stripping, no
  greedy regex, no brace walking.
- Pydantic-validated output is a dict by construction — no
  "instruction_form returned no output_data" silent failures.
- Decision nodes pick from an ``enum`` — the server can only emit a
  member of the enum, so the fuzzy ``opt.lower() in picked.lower()``
  matcher isn't needed for the happy path.

The nodes still fall back to the old free-form path when a provider
ignores ``tool_choice`` or the model skips the tool — no regression on
older local setups.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from quartermaster_providers import LLMConfig, ProviderRegistry
from quartermaster_providers.providers.openai import OpenAIProvider
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse, ToolCall

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.example_runner import (
    DecisionExecutor,
    InstructionFormExecutor,
)
from quartermaster_engine.types import (
    GraphEdge,
    GraphNode,
    GraphSpec,
    NodeType,
)

# ── Unit: LLMConfig.tool_choice plumbs through OpenAIProvider ────────


class TestToolChoicePassthrough:
    """``LLMConfig(tool_choice=...)`` must land in the outgoing
    ``chat.completions.create(..., tool_choice=<value>)`` call."""

    def _mock_client(self, content: str = "ok") -> Any:
        from unittest.mock import AsyncMock, MagicMock

        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=resp)
        return client

    def test_dict_tool_choice_forwarded(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        client = self._mock_client()
        provider._client = client

        config = LLMConfig(
            model="gpt-4o",
            provider="openai",
            tool_choice={
                "type": "function",
                "function": {"name": "emit_result"},
            },
        )
        asyncio.run(provider.generate_text_response("hi", config))

        seen = client.chat.completions.create.call_args.kwargs
        assert seen.get("tool_choice") == {
            "type": "function",
            "function": {"name": "emit_result"},
        }

    def test_string_tool_choice_forwarded(self) -> None:
        """``tool_choice="required"`` / ``"none"`` / ``"auto"`` are the
        shorthand forms openai accepts — same passthrough path."""
        provider = OpenAIProvider(api_key="sk-test")
        client = self._mock_client()
        provider._client = client

        config = LLMConfig(model="gpt-4o", provider="openai", tool_choice="required")
        asyncio.run(provider.generate_text_response("hi", config))

        seen = client.chat.completions.create.call_args.kwargs
        assert seen.get("tool_choice") == "required"

    def test_absent_when_not_set(self) -> None:
        """No ``tool_choice`` in the request when none was configured —
        the openai SDK's default behaviour must apply."""
        provider = OpenAIProvider(api_key="sk-test")
        client = self._mock_client()
        provider._client = client

        config = LLMConfig(model="gpt-4o", provider="openai")
        asyncio.run(provider.generate_text_response("hi", config))

        seen = client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in seen


# ── Integration helpers ──────────────────────────────────────────────


def _make_ctx_form(
    schema_json: str | None = None,
    schema_class: str | None = None,
) -> ExecutionContext:
    metadata: dict[str, Any] = {"llm_provider": "mock", "llm_model": "mock-m"}
    if schema_json:
        metadata["schema_json"] = schema_json
    if schema_class:
        metadata["schema_class"] = schema_class
    node = GraphNode(
        id=uuid4(),
        type=NodeType.INSTRUCTION_FORM,
        name="ExtractCompany",
        metadata=metadata,
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
        memory={"__user_input__": "Enrich Makro Mikro"},
        metadata={},
    )


def _make_ctx_decision(option_labels: list[str]) -> ExecutionContext:
    node = GraphNode(
        id=uuid4(),
        type=NodeType.DECISION,
        name="Router",
        metadata={"llm_provider": "mock", "llm_model": "mock-m"},
    )
    target_nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for label in option_labels:
        target = GraphNode(id=uuid4(), type=NodeType.INSTRUCTION, name=label, metadata={})
        target_nodes.append(target)
        edges.append(GraphEdge(source_id=node.id, target_id=target.id, label=label))
    graph = GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=node.id,
        nodes=[node, *target_nodes],
        edges=edges,
    )
    return ExecutionContext(
        flow_id=uuid4(),
        node_id=node.id,
        graph=graph,
        current_node=node,
        messages=[],
        memory={"__user_input__": "Route me"},
        metadata={},
    )


def _registry_with(mock: MockProvider) -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register_instance("mock", mock)
    return reg


# ── Integration: InstructionFormExecutor uses forced tool-call ───────


class TestInstructionFormForcedToolCall:
    """The forced-tool-call happy path — provider returned structured
    ``tool_calls[0].parameters`` so no text parsing runs at all."""

    def test_tool_call_params_become_parsed_output(self) -> None:
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="emit_result",
                            tool_id="call_0",
                            parameters={
                                "city": "Zagreb",
                                "country": "HR",
                                "industry": "construction",
                            },
                        )
                    ],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx_form(
            schema_json=(
                '{"type": "object", "properties": '
                '{"city": {"type": "string"}, "country": {"type": "string"}, '
                '"industry": {"type": "string"}}}'
            )
        )
        executor = InstructionFormExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        assert result.success, result.error
        assert result.data["parsed"] == {
            "city": "Zagreb",
            "country": "HR",
            "industry": "construction",
        }

    def test_config_tool_choice_set_to_emit_result(self) -> None:
        """The executor must set ``config.tool_choice`` to force the
        ``emit_result`` tool — otherwise vLLM's auto tool-choice could
        skip the call and fall back to free-form text."""
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="emit_result",
                            tool_id="call_0",
                            parameters={"ok": True},
                        )
                    ],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx_form(
            schema_json='{"type": "object", "properties": {"ok": {"type": "boolean"}}}'
        )
        executor = InstructionFormExecutor(_registry_with(mock))

        asyncio.run(executor.execute(ctx))

        assert mock.last_config is not None
        assert mock.last_config.tool_choice == {
            "type": "function",
            "function": {"name": "emit_result"},
        }

    def test_falls_back_to_text_path_when_provider_ignores_tool_choice(self) -> None:
        """Older models / providers without forced tool-choice support
        still get a free-form response — the executor's text-parse
        fallback (v0.6.3 three-tier walker) must still fire."""
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content='{"city": "Zagreb", "country": "HR"}',
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx_form(
            schema_json=(
                '{"type": "object", "properties": '
                '{"city": {"type": "string"}, "country": {"type": "string"}}}'
            )
        )
        executor = InstructionFormExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        assert result.success, result.error
        assert result.data["parsed"] == {"city": "Zagreb", "country": "HR"}

    def test_no_schema_skips_tool_choice_entirely(self) -> None:
        """When the node has no schema, we mustn't set ``tool_choice``
        — the instruction_form still runs in free-text mode for nodes
        that just want an LLM turn without a shape."""
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content='{"anything": "goes"}',
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx_form()  # no schema_json
        executor = InstructionFormExecutor(_registry_with(mock))

        asyncio.run(executor.execute(ctx))

        assert mock.last_config is not None
        assert mock.last_config.tool_choice is None


# ── Integration: DecisionExecutor uses forced tool-call ──────────────


class TestDecisionForcedToolCall:
    """The forced-tool-call happy path for decision nodes. Before v0.7.0
    decisions used free-form + fuzzy matching: ``"Respond with EXACTLY
    one of the options"`` → ``opt.lower() in picked.lower()``. The LLM
    could still drift and we'd misroute. Now the server-side enum
    constraint on the ``choice`` param eliminates that risk for
    provider-supported paths."""

    def test_tool_call_choice_becomes_picked_node(self) -> None:
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="pick_branch",
                            tool_id="call_0",
                            parameters={"choice": "approve"},
                        )
                    ],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx_decision(["approve", "reject", "escalate"])
        executor = DecisionExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        assert result.success
        assert result.picked_node == "approve"

    def test_config_tool_choice_forces_pick_branch(self) -> None:
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="pick_branch",
                            tool_id="call_0",
                            parameters={"choice": "approve"},
                        )
                    ],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx_decision(["approve", "reject"])
        executor = DecisionExecutor(_registry_with(mock))

        asyncio.run(executor.execute(ctx))

        assert mock.last_config is not None
        assert mock.last_config.tool_choice == {
            "type": "function",
            "function": {"name": "pick_branch"},
        }

    def test_choice_not_in_enum_falls_back_to_fuzzy_match(self) -> None:
        """If the model somehow emits a ``choice`` outside the enum
        (e.g. off-spec provider), we fall through to the fuzzy
        text-matching path rather than raising."""
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="I'd go with the approve option.",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="pick_branch",
                            tool_id="call_0",
                            parameters={"choice": "APPROVE_WITH_NOTES"},
                        )
                    ],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx_decision(["approve", "reject"])
        executor = DecisionExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        # Fuzzy fallback on the text_content picks up ``approve``.
        assert result.picked_node == "approve"

    def test_falls_back_to_free_form_text_when_no_tool_call(self) -> None:
        """Provider that didn't honour tool_choice and returned free
        text — the fuzzy match on the text_content still picks a
        branch. This is the legacy path."""
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="reject",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                )
            ],
            responses=[TokenResponse(content="reject", stop_reason="stop")],
        )
        ctx = _make_ctx_decision(["approve", "reject"])
        executor = DecisionExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        assert result.picked_node == "reject"

    def test_no_options_skips_tool_choice(self) -> None:
        """A decision node with no outgoing labels has no enum to
        constrain on — we mustn't send a broken tool definition."""
        mock = MockProvider(
            responses=[TokenResponse(content="", stop_reason="stop")],
        )
        ctx = _make_ctx_decision([])
        executor = DecisionExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        assert result.success
        # No options → no tool_choice — legacy free-text path.
        assert mock.last_config is None or mock.last_config.tool_choice is None
