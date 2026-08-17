"""v0.9.0 — LLMExecutor / AgentExecutor put provider usage on NodeResult (#100)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse, TokenUsage

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.example_runner import AgentExecutor, LLMExecutor
from quartermaster_engine.types import GraphNode, GraphSpec, NodeType


def _ctx(node_type: NodeType, name: str, provider: str, model: str) -> ExecutionContext:
    node = GraphNode(
        id=uuid4(),
        type=node_type,
        name=name,
        metadata={"llm_provider": provider, "llm_model": model},
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
        memory={"__user_input__": "hello"},
        metadata={},
    )


def _registry(mock: MockProvider) -> ProviderRegistry:
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("ollama", mock)
    reg.set_default_provider("ollama")
    reg.set_default_model("ollama", "mock-model")
    return reg


class TestLLMExecutorUsage:
    def test_instruction_node_data_includes_usage(self) -> None:
        usage = TokenUsage(input_tokens=15, output_tokens=6)
        mock = MockProvider(
            responses=[TokenResponse(content="hi there", stop_reason="stop", usage=usage)]
        )
        ctx = _ctx(NodeType.INSTRUCTION, "Instruction", "ollama", "mock-model")
        result = asyncio.run(LLMExecutor(_registry(mock)).execute(ctx))

        assert result.success, result.error
        assert result.output_text == "hi there"
        assert result.data["usage"] == {"input_tokens": 15, "output_tokens": 6}

    def test_instruction_node_omits_usage_when_provider_does(self) -> None:
        mock = MockProvider(responses=[TokenResponse(content="hi there", stop_reason="stop")])
        ctx = _ctx(NodeType.INSTRUCTION, "Instruction", "ollama", "mock-model")
        result = asyncio.run(LLMExecutor(_registry(mock)).execute(ctx))

        assert result.success, result.error
        assert "usage" not in result.data


class TestAgentExecutorUsage:
    def test_agent_node_data_includes_native_usage(self) -> None:
        usage = TokenUsage(input_tokens=20, output_tokens=8)
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="done",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                    usage=usage,
                )
            ]
        )
        ctx = _ctx(NodeType.AGENT, "Agent", "ollama", "mock-model")
        result = asyncio.run(AgentExecutor(_registry(mock)).execute(ctx))

        assert result.success, result.error
        assert result.data["usage"] == {"input_tokens": 20, "output_tokens": 8}

    def test_agent_node_omits_usage_when_provider_does(self) -> None:
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="done",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                    usage=None,
                )
            ]
        )
        ctx = _ctx(NodeType.AGENT, "Agent", "ollama", "mock-model")
        result = asyncio.run(AgentExecutor(_registry(mock)).execute(ctx))

        assert result.success, result.error
        assert "usage" not in result.data
